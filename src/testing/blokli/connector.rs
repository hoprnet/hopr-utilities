//! A minimal chain connector backed by a Blokli query/subscription/transaction client, for use in
//! unit and integration tests.

use std::{
    str::FromStr,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use blokli_client::api::{
    AccountSelector as BlokliAccountSelector, BlokliQueryClient, BlokliSubscriptionClient, BlokliTransactionClient,
    ChannelSelector, RedeemedStatsSelector, SafeSelector as BlokliSafeSelector,
};
use futures::StreamExt;
use hopr_api::types::{
    chain::prelude::{PayloadGenerator, SignableTransaction},
    crypto::{
        prelude::{Keypair, OffchainKeypair},
        types::Hash,
    },
    internal::prelude::{
        AccountEntry, AccountType, ChannelBuilder, ChannelId, ChannelStatus, RedeemableTicket, VerifiedTicket,
        WinningProbability, generate_channel_id,
    },
    primitive::prelude::{Address, HoprBalance, WxHOPR, XDai},
};

use super::faults::{ChainFaults, ChainOp, Fault};

/// Concrete error type used by [`TestChainConnector`] trait implementations.
///
/// The chain API traits require `Error: std::error::Error + Send + Sync + 'static`.
/// `anyhow::Error` does not implement `std::error::Error` directly, so this
/// transparent newtype bridges the gap.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct TestConnectorError(#[from] anyhow::Error);

impl From<blokli_client::errors::BlokliClientError> for TestConnectorError {
    fn from(e: blokli_client::errors::BlokliClientError) -> Self {
        Self(anyhow::anyhow!("{e}"))
    }
}

impl From<hopr_api::types::primitive::prelude::GeneralError> for TestConnectorError {
    fn from(e: hopr_api::types::primitive::prelude::GeneralError) -> Self {
        Self(anyhow::anyhow!("{e}"))
    }
}

impl From<hopr_api::types::chain::errors::ChainTypesError> for TestConnectorError {
    fn from(e: hopr_api::types::chain::errors::ChainTypesError) -> Self {
        Self(anyhow::anyhow!("{e}"))
    }
}

impl From<hopr_api::types::internal::prelude::CoreTypesError> for TestConnectorError {
    fn from(e: hopr_api::types::internal::prelude::CoreTypesError) -> Self {
        Self(anyhow::anyhow!("{e}"))
    }
}

/// A noop key mapper that always returns `None` for all key lookups.
///
/// This is sufficient for unit tests that do not exercise the
/// packet-key/chain-key mapping code path.
#[derive(Clone, Debug, Default)]
pub struct NoopKeyMapper;

impl hopr_api::chain::KeyIdMapping<hopr_api::chain::HoprKeyIdent, hopr_api::types::crypto::prelude::OffchainPublicKey>
    for NoopKeyMapper
{
    fn map_key_to_id(
        &self,
        _key: &hopr_api::types::crypto::prelude::OffchainPublicKey,
    ) -> Option<hopr_api::chain::HoprKeyIdent> {
        None
    }

    fn map_id_to_public(
        &self,
        _id: &hopr_api::chain::HoprKeyIdent,
    ) -> Option<hopr_api::types::crypto::prelude::OffchainPublicKey> {
        None
    }
}

type TestEventsChannel = (
    async_broadcast::Sender<hopr_api::chain::ChainEvent>,
    async_broadcast::InactiveReceiver<hopr_api::chain::ChainEvent>,
);

struct ParsedChainInfo {
    chain_info: hopr_api::chain::ChainInfo,
    domain_separators: hopr_api::chain::DomainSeparators,
    ticket_win_prob: WinningProbability,
    ticket_price: HoprBalance,
    closure_grace_period: std::time::Duration,
}

/// A minimal chain connector backed by a Blokli client for use in unit tests.
///
/// Wraps a Blokli query/subscription/transaction client and implements all
/// [`HoprChainApi`](hopr_api::chain::HoprChainApi) sub-traits with `Error = anyhow::Error`. Write
/// operations that unit tests do not exercise return an error instead of panicking.
pub struct TestChainConnector<C> {
    client: Arc<C>,
    my_addr: Address,
    chain_key: hopr_api::types::crypto::prelude::ChainKeypair,
    module_address: Address,
    events: TestEventsChannel,
    /// Payload generator, initialized on `connect()` after fetching chain info.
    payload_gen: OnceLock<hopr_api::types::chain::payload::SafePayloadGenerator>,
    /// chain_id, initialized on `connect()`.
    chain_id: OnceLock<u64>,
    /// Ticket price from chain info, populated on `connect()` for synchronous access.
    ticket_price: OnceLock<HoprBalance>,
    /// Minimum winning probability from chain info, populated on `connect()` for synchronous access.
    ticket_win_prob: OnceLock<WinningProbability>,
    /// Nonce counters for transaction sequencing, one per signing address.
    ///
    /// Keyed by signer because `withdraw_from_signer` signs with a caller-supplied key: a
    /// single shared counter would hand that key a nonce advanced by the connector's own
    /// transactions, and then reuse a nonce for the connector.
    nonces: Arc<dashmap::DashMap<Address, Arc<AtomicU64>>>,
    /// Accounts cache: chain address → AccountEntry, populated on `connect()`.
    accounts: Arc<dashmap::DashMap<Address, AccountEntry>>,
    /// Channel cache: channel id → ChannelEntry, populated on `connect()`.
    channels: Arc<dashmap::DashMap<ChannelId, hopr_api::types::internal::prelude::ChannelEntry>>,
    /// Injected chain-operation faults; empty unless a test configures them.
    faults: Arc<ChainFaults>,
}

impl<C> TestChainConnector<C>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    /// Creates a disconnected connector over `client`.
    ///
    /// Call [`TestChainConnector::connect`] before any read operation; before that, chain info,
    /// ticket values, and the payload generator are unset.
    pub fn new(
        client: C,
        my_addr: Address,
        chain_key: hopr_api::types::crypto::prelude::ChainKeypair,
        module_address: Address,
    ) -> Self {
        let (mut tx, rx) = async_broadcast::broadcast(256);
        tx.set_overflow(true);
        Self {
            client: Arc::new(client),
            my_addr,
            chain_key,
            module_address,
            events: (tx, rx.deactivate()),
            payload_gen: Default::default(),
            chain_id: Default::default(),
            ticket_price: Default::default(),
            ticket_win_prob: Default::default(),
            nonces: Default::default(),
            accounts: Default::default(),
            channels: Default::default(),
            faults: Default::default(),
        }
    }

    /// Handle to this connector's fault configuration.  Faults can be set and
    /// cleared at any time, including while a strategy is running against it.
    pub fn faults(&self) -> Arc<ChainFaults> {
        self.faults.clone()
    }

    /// A handle to the same in-process chain this connector talks to.
    ///
    /// The underlying client shares its state when cloned, so the returned handle sees and
    /// mutates exactly the state this connector does. Needed by anything that builds a *second*
    /// connector over the same chain — a component signing as an EOA rather than through the
    /// node's Safe, say — which otherwise has no way to reach it.
    pub fn client(&self) -> C {
        (*self.client).clone()
    }

    /// Loads initial state via finite queries and spawns a background task for live event forwarding.
    pub async fn connect(&mut self) -> anyhow::Result<()> {
        // Fetch chain info to initialize the payload generator and cache ticket values.
        let chain_info_raw = self.client.query_chain_info().await?;
        let parsed = Self::parse_chain_info_model(chain_info_raw)?;
        let hopr_api::chain::ChainInfo {
            chain_id,
            contract_addresses,
            ..
        } = parsed.chain_info;
        let _ = self.chain_id.set(chain_id);
        let _ = self
            .payload_gen
            .set(hopr_api::types::chain::payload::SafePayloadGenerator::new(
                &self.chain_key,
                contract_addresses,
                self.module_address,
            ));
        let _ = self.ticket_price.set(parsed.ticket_price);
        let _ = self.ticket_win_prob.set(parsed.ticket_win_prob);

        // Load all accounts via a finite snapshot query and build a keyid→address map.
        let mut keyid_to_addr = std::collections::HashMap::<u32, Address>::new();
        for account_model in self.client.query_accounts(BlokliAccountSelector::Any).await? {
            let entry = Self::convert_account_model(account_model)?;
            keyid_to_addr.insert(u32::from(entry.key_id), entry.chain_addr);
            self.accounts.insert(entry.chain_addr, entry);
        }

        // Load all channels via a finite snapshot query and fire ChannelOpened for Open channels.
        for channel_model in self.client.query_channels(ChannelSelector::default()).await?.channels {
            let src_addr = keyid_to_addr
                .get(&(channel_model.source as u32))
                .copied()
                .ok_or_else(|| anyhow::anyhow!("source key_id {} not found in accounts", channel_model.source))?;
            let dst_addr = keyid_to_addr
                .get(&(channel_model.destination as u32))
                .copied()
                .ok_or_else(|| {
                    anyhow::anyhow!("destination key_id {} not found in accounts", channel_model.destination)
                })?;
            let channel = Self::convert_channel_model(&channel_model, src_addr, dst_addr)?;
            let channel_id = *channel.get_id();
            self.channels.insert(channel_id, channel);
        }

        // Spawn a background task that forwards live graph updates as ChainEvents.
        // subscribe_graph() emits an initial snapshot (already loaded above) then live updates.
        // The initial snapshot items produce no-op comparisons against the cache; only real
        // state changes trigger new events.
        let client = self.client.clone();
        let events_tx = self.events.0.clone();
        let accounts_cache = self.accounts.clone();
        let channels_cache = self.channels.clone();
        crate::runtime::prelude::spawn(async move {
            use futures::TryStreamExt;

            let graph_stream = match client.subscribe_graph() {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("subscribe_graph() failed in background event loop: {e}");
                    return;
                }
            };
            futures::pin_mut!(graph_stream);

            loop {
                let entry = match graph_stream.try_next().await {
                    Ok(Some(e)) => e,
                    Ok(None) => break,
                    Err(e) => {
                        tracing::error!("graph stream error in background event loop: {e}");
                        break;
                    }
                };
                let src = match Self::convert_account_model(entry.source) {
                    Ok(a) => a,
                    Err(e) => {
                        tracing::warn!("failed to convert account in graph event: {e}");
                        continue;
                    }
                };
                let dst = match Self::convert_account_model(entry.destination) {
                    Ok(a) => a,
                    Err(e) => {
                        tracing::warn!("failed to convert account in graph event: {e}");
                        continue;
                    }
                };
                let new_channel = match Self::convert_channel_model(&entry.channel, src.chain_addr, dst.chain_addr) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!("failed to convert channel in graph event: {e}");
                        continue;
                    }
                };

                accounts_cache.insert(src.chain_addr, src);
                accounts_cache.insert(dst.chain_addr, dst);

                let channel_id = *new_channel.get_id();
                let old_channel = channels_cache.get(&channel_id).map(|r| *r);
                channels_cache.insert(channel_id, new_channel);

                let event = match old_channel {
                    None => {
                        if new_channel.status == ChannelStatus::Open {
                            Some(hopr_api::chain::ChainEvent::ChannelOpened(new_channel))
                        } else {
                            None
                        }
                    }
                    Some(ref old) if old.status == new_channel.status && old.balance == new_channel.balance => None,
                    Some(ref old) if old.status != new_channel.status => match new_channel.status {
                        ChannelStatus::Open => Some(hopr_api::chain::ChainEvent::ChannelOpened(new_channel)),
                        ChannelStatus::PendingToClose(_) => {
                            Some(hopr_api::chain::ChainEvent::ChannelClosureInitiated(new_channel))
                        }
                        ChannelStatus::Closed => Some(hopr_api::chain::ChainEvent::ChannelClosed(new_channel)),
                    },
                    Some(ref old) => {
                        if new_channel.balance > old.balance {
                            let diff = new_channel.balance - old.balance;
                            Some(hopr_api::chain::ChainEvent::ChannelBalanceIncreased(new_channel, diff))
                        } else if new_channel.ticket_index > old.ticket_index {
                            Some(hopr_api::chain::ChainEvent::TicketRedeemed(new_channel, None))
                        } else {
                            let diff = old.balance - new_channel.balance;
                            Some(hopr_api::chain::ChainEvent::ChannelBalanceDecreased(new_channel, diff))
                        }
                    }
                };

                if let Some(evt) = event {
                    let _ = events_tx.try_broadcast(evt);
                }
            }
        });

        Ok(())
    }

    fn convert_account_model(model: blokli_client::api::types::Account) -> anyhow::Result<AccountEntry> {
        let entry_type = if !model.multi_addresses.is_empty() {
            AccountType::Announced(
                model
                    .multi_addresses
                    .into_iter()
                    .filter_map(|a| hopr_api::chain::Multiaddr::from_str(&a).ok())
                    .collect(),
            )
        } else {
            AccountType::NotAnnounced
        };

        Ok(AccountEntry {
            public_key: model.packet_key.parse()?,
            chain_addr: model.chain_key.parse()?,
            key_id: (model.keyid as u32).into(),
            entry_type,
            safe_address: model.safe_address.map(|a| a.parse::<Address>()).transpose()?,
        })
    }

    fn convert_channel_model(
        model: &blokli_client::api::types::Channel,
        src_addr: Address,
        dst_addr: Address,
    ) -> anyhow::Result<hopr_api::types::internal::prelude::ChannelEntry> {
        let status = match model.status {
            blokli_client::api::types::ChannelStatus::Open => ChannelStatus::Open,
            blokli_client::api::types::ChannelStatus::PendingToClose => {
                let closure_time = model
                    .closure_time
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing closure time on PendingToClose channel"))?;
                ChannelStatus::PendingToClose(hopr_api::chain::DateTime::from_str(&closure_time.0)?.into())
            }
            blokli_client::api::types::ChannelStatus::Closed => ChannelStatus::Closed,
        };

        Ok(ChannelBuilder::default()
            .between(src_addr, dst_addr)
            .balance(model.balance.0.parse()?)
            .ticket_index(
                model
                    .ticket_index
                    .0
                    .parse()
                    .map_err(|e| anyhow::anyhow!("invalid ticket index: {e}"))?,
            )
            .status(status)
            .epoch(model.epoch as u32)
            .build()?)
    }

    /// Nonce counter for `signer`, created on first use.
    ///
    /// Always pair this with the key that actually signs the transaction — see the note on
    /// [`TestChainConnector::nonces`].
    fn nonce_for(&self, signer: &Address) -> Arc<AtomicU64> {
        self.nonces.entry(*signer).or_default().clone()
    }

    /// Waits until this connector's own channel view satisfies `predicate`.
    ///
    /// The emulated RPC wraps the chain, not the other way round: the tx has
    /// already executed and broadcast by the time submission returns, so a
    /// confirmation must not resolve before the caller can read the result.
    /// This connector ingests that broadcast on a background task, so without
    /// the wait it would report success against a view it has not caught up
    /// with — which no real chain RPC does.
    async fn await_own_view(
        channels: Arc<dashmap::DashMap<ChannelId, hopr_api::types::internal::prelude::ChannelEntry>>,
        channel_id: ChannelId,
        predicate: impl Fn(&hopr_api::types::internal::prelude::ChannelEntry) -> bool,
    ) {
        // Generous for an in-process broadcast: only a stuck background task
        // reaches it, and the caller's own timeout covers that.
        const LIMIT: std::time::Duration = std::time::Duration::from_secs(5);
        const POLL: std::time::Duration = std::time::Duration::from_millis(2);

        let deadline = std::time::Instant::now() + LIMIT;
        loop {
            if channels.get(&channel_id).is_some_and(|entry| predicate(&entry)) {
                return;
            }
            if std::time::Instant::now() >= deadline {
                tracing::warn!(%channel_id, "test connector: own view never caught up with the confirmed tx");
                return;
            }
            futures_time::task::sleep(POLL.into()).await;
        }
    }

    async fn send_tx(
        client: &C,
        tx_req: hopr_api::types::chain::payload::TransactionRequest,
        chain_id: u64,
        chain_key: &hopr_api::types::crypto::prelude::ChainKeypair,
        nonce: &AtomicU64,
    ) -> anyhow::Result<hopr_api::chain::ChainReceipt> {
        let n = nonce.fetch_add(1, Ordering::Relaxed);
        let signed = tx_req.sign_and_encode_to_eip2718(n, chain_id, None, chain_key).await?;
        let receipt = client.submit_and_confirm_transaction(&signed, 1).await?;
        Ok(hopr_api::chain::ChainReceipt::from(receipt))
    }

    fn parse_chain_info_model(model: blokli_client::api::types::ChainInfo) -> anyhow::Result<ParsedChainInfo> {
        let channel_closure_grace_period = std::time::Duration::from_secs(
            model
                .channel_closure_grace_period
                .0
                .parse()
                .map_err(|e| anyhow::anyhow!("invalid closure grace period: {e}"))?,
        );

        let domain_separators = hopr_api::chain::DomainSeparators {
            ledger: model
                .ledger_dst
                .as_deref()
                .map(Hash::from_str)
                .transpose()?
                .unwrap_or_default(),
            safe_registry: model
                .safe_registry_dst
                .as_deref()
                .map(Hash::from_str)
                .transpose()?
                .unwrap_or_default(),
            channel: model
                .channel_dst
                .as_deref()
                .map(Hash::from_str)
                .transpose()?
                .unwrap_or_default(),
        };

        let ticket_win_prob = WinningProbability::try_from_f64(model.min_ticket_winning_probability)?;
        let ticket_price: HoprBalance = model.ticket_price.0.parse()?;
        let chain_info = hopr_api::chain::ChainInfo {
            chain_id: model.chain_id as u64,
            hopr_network_name: model.network,
            contract_addresses: serde_json::from_str(&model.contract_addresses.0)
                .map_err(|e| anyhow::anyhow!("invalid contract addresses: {e}"))?,
        };

        Ok(ParsedChainInfo {
            chain_info,
            domain_separators,
            ticket_win_prob,
            ticket_price,
            closure_grace_period: channel_closure_grace_period,
        })
    }

    fn payload_gen(&self) -> anyhow::Result<&hopr_api::types::chain::payload::SafePayloadGenerator> {
        self.payload_gen
            .get()
            .ok_or_else(|| anyhow::anyhow!("connector not connected"))
    }

    fn chain_id(&self) -> anyhow::Result<u64> {
        self.chain_id
            .get()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("connector not connected"))
    }

    /// Fetches and parses live chain info.
    ///
    /// Deliberately not cached: some tests mutate the emulated chain's `chain_info` mid-run (e.g.
    /// to simulate a chain that starts returning unparseable data), and every [`ChainValues`]
    /// getter built on this must observe that change on its next call.
    ///
    /// [`ChainValues`]: hopr_api::chain::ChainValues
    async fn fetch_parsed_chain_info(&self) -> anyhow::Result<ParsedChainInfo> {
        let model = self.client.query_chain_info().await?;
        Self::parse_chain_info_model(model)
    }

    /// Shared body of [`ChainWriteAccountOperations::withdraw`] and `::withdraw_from_signer`:
    /// builds and submits a transfer signed by `signer`, using that signer's own nonce sequence.
    ///
    /// [`ChainWriteAccountOperations::withdraw`]: hopr_api::chain::ChainWriteAccountOperations::withdraw
    fn withdraw_as<'a, Cur: hopr_api::types::primitive::prelude::Currency + Send>(
        &'a self,
        signer: &hopr_api::types::crypto::prelude::ChainKeypair,
        balance: hopr_api::types::primitive::prelude::Balance<Cur>,
        recipient: &Address,
    ) -> Result<
        futures::future::BoxFuture<'a, Result<hopr_api::chain::ChainReceipt, TestConnectorError>>,
        TestConnectorError,
    > {
        let tx_req = self
            .payload_gen()
            .map_err(TestConnectorError::from)?
            .transfer(*recipient, balance)
            .map_err(|e| TestConnectorError::from(anyhow::anyhow!("{e}")))?;

        let client = self.client.clone();
        let chain_id = self.chain_id().map_err(TestConnectorError::from)?;
        let nonce = self.nonce_for(&signer.public().to_address());
        let signer = signer.clone();

        Ok(Box::pin(async move {
            Self::send_tx(&client, tx_req, chain_id, &signer, &nonce)
                .await
                .map_err(TestConnectorError::from)
        }))
    }

    /// Shared tail of [`ChainWriteChannelOperations`]'s three write methods: waits for this
    /// connector's own view to reflect the just-submitted change, then resolves any injected
    /// confirmation fault before yielding `receipt`.
    ///
    /// [`ChainWriteChannelOperations`]: hopr_api::chain::ChainWriteChannelOperations
    fn track_confirmation<'a>(
        &'a self,
        op: ChainOp,
        channel_id: ChannelId,
        receipt: hopr_api::chain::ChainReceipt,
        predicate: impl Fn(&hopr_api::types::internal::prelude::ChannelEntry) -> bool + Send + 'static,
    ) -> futures::future::BoxFuture<'a, Result<hopr_api::chain::ChainReceipt, TestConnectorError>> {
        let faults = self.faults.clone();
        let channels = self.channels.clone();
        let in_flight = faults.enter_in_flight(op);
        Box::pin(async move {
            let _in_flight = in_flight;
            Self::await_own_view(channels, channel_id, predicate).await;
            faults.confirm(op).await?;
            Ok(receipt)
        })
    }
}

// ── ChainReadAccountOperations ────────────────────────────────────────────────

#[async_trait::async_trait]
impl<C> hopr_api::chain::ChainReadAccountOperations for TestChainConnector<C>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    type Error = TestConnectorError;

    fn stream_accounts<'a>(
        &'a self,
        selector: hopr_api::chain::AccountSelector,
    ) -> Result<futures::stream::BoxStream<'a, AccountEntry>, Self::Error> {
        if self.faults.gate_stream(ChainOp::StreamAccounts)? == Fault::Hang {
            return Ok(futures::stream::pending().boxed());
        }

        let entries: Vec<_> = self
            .accounts
            .iter()
            .filter(|e| selector.satisfies(e.value()))
            .map(|e| e.value().clone())
            .collect();
        Ok(futures::stream::iter(entries).boxed())
    }

    async fn count_accounts(&self, selector: hopr_api::chain::AccountSelector) -> Result<usize, Self::Error> {
        Ok(self.accounts.iter().filter(|e| selector.satisfies(e.value())).count())
    }

    async fn await_key_binding(
        &self,
        offchain_key: &hopr_api::types::crypto::prelude::OffchainPublicKey,
        _timeout: std::time::Duration,
    ) -> Result<AccountEntry, Self::Error> {
        self.accounts
            .iter()
            .find(|e| &e.value().public_key == offchain_key)
            .map(|e| e.value().clone())
            .ok_or_else(|| TestConnectorError::from(anyhow::anyhow!("account with key {offchain_key} not found")))
    }
}

// ── ChainWriteAccountOperations ───────────────────────────────────────────────

#[async_trait::async_trait]
impl<C> hopr_api::chain::ChainWriteAccountOperations for TestChainConnector<C>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    type Error = TestConnectorError;

    async fn announce(
        &self,
        _multiaddrs: &[hopr_api::chain::Multiaddr],
        _key: &OffchainKeypair,
    ) -> Result<
        futures::future::BoxFuture<'_, Result<hopr_api::chain::ChainReceipt, Self::Error>>,
        hopr_api::chain::AnnouncementError<Self::Error>,
    > {
        Err(hopr_api::chain::AnnouncementError::processing(anyhow::anyhow!(
            "not supported by TestChainConnector"
        )))
    }

    async fn withdraw<Cur: hopr_api::types::primitive::prelude::Currency + Send>(
        &self,
        balance: hopr_api::types::primitive::prelude::Balance<Cur>,
        recipient: &Address,
    ) -> Result<futures::future::BoxFuture<'_, Result<hopr_api::chain::ChainReceipt, Self::Error>>, Self::Error> {
        self.withdraw_as(&self.chain_key, balance, recipient)
    }

    async fn withdraw_from_signer<Cur: hopr_api::types::primitive::prelude::Currency + Send>(
        &self,
        signer: &hopr_api::types::crypto::prelude::ChainKeypair,
        balance: hopr_api::types::primitive::prelude::Balance<Cur>,
        recipient: &Address,
    ) -> Result<futures::future::BoxFuture<'_, Result<hopr_api::chain::ChainReceipt, Self::Error>>, Self::Error> {
        self.withdraw_as(signer, balance, recipient)
    }

    async fn register_safe(
        &self,
        safe_address: &Address,
    ) -> Result<
        futures::future::BoxFuture<'_, Result<hopr_api::chain::ChainReceipt, Self::Error>>,
        hopr_api::chain::SafeRegistrationError<Self::Error>,
    > {
        let my_addr = self.my_addr;

        // Check if already registered
        if let Some(existing) = self
            .client
            .query_safe(BlokliSafeSelector::RegisteredNode(my_addr.into()))
            .await
            .map_err(hopr_api::chain::SafeRegistrationError::processing)?
            .first()
        {
            let registered = existing
                .address
                .parse::<Address>()
                .map_err(hopr_api::chain::SafeRegistrationError::processing)?;
            return Err(hopr_api::chain::SafeRegistrationError::AlreadyRegistered(registered));
        }

        // Check the safe exists
        if self
            .client
            .query_safe(BlokliSafeSelector::SafeAddress((*safe_address).into()))
            .await
            .map_err(hopr_api::chain::SafeRegistrationError::processing)?
            .is_empty()
        {
            return Err(hopr_api::chain::SafeRegistrationError::processing(anyhow::anyhow!(
                "safe {safe_address} does not exist"
            )));
        }

        let tx_req = self
            .payload_gen()
            .map_err(hopr_api::chain::SafeRegistrationError::processing)?
            .register_safe_by_node(*safe_address)
            .map_err(hopr_api::chain::SafeRegistrationError::processing)?;

        let client = self.client.clone();
        let chain_id = self
            .chain_id()
            .map_err(hopr_api::chain::SafeRegistrationError::processing)?;
        let chain_key = self.chain_key.clone();
        let nonce = self.nonce_for(&self.my_addr);
        Ok(Box::pin(async move {
            Self::send_tx(&client, tx_req, chain_id, &chain_key, &nonce)
                .await
                .map_err(TestConnectorError::from)
        }))
    }
}

// ── Service registry ─────────────────────────────────────────────────────────
//
// `HoprChainApi` gained the two service-registry traits, so a chain API has to answer for them
// even though no strategy in the consuming crate reads or writes the registry — none is about
// services. The Blokli emulator behind this connector does not model the registry at all, so
// these are stubs rather than a thin layer over it.
//
// Reads answer as an *empty* registry: that is a truthful answer for a chain where nothing was
// ever registered, and it keeps a caller that merely surveys the registry working. Writes report
// "not supported", matching `announce` above — silently accepting a registration the emulator
// cannot store would make a later read look like a lost write.

#[async_trait::async_trait]
impl<C> hopr_api::chain::ChainReadServiceOperations for TestChainConnector<C>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    type Error = TestConnectorError;

    fn stream_services(
        &self,
        _selector: hopr_api::chain::ServiceSelector,
    ) -> Result<futures::stream::BoxStream<'_, hopr_api::chain::ServiceEntry>, Self::Error> {
        Ok(futures::stream::empty().boxed())
    }

    async fn count_services(&self, _selector: hopr_api::chain::ServiceSelector) -> Result<usize, Self::Error> {
        Ok(0)
    }

    async fn get_service_type_config(
        &self,
        _service_type: hopr_api::chain::ServiceType,
    ) -> Result<Option<hopr_api::chain::ServiceTypeConfig>, Self::Error> {
        Ok(None)
    }

    async fn get_service_registry_config(&self) -> Result<hopr_api::chain::ServiceRegistryConfig, Self::Error> {
        Ok(hopr_api::chain::ServiceRegistryConfig {
            type_registration_fee: HoprBalance::zero(),
            node_safe_registry: Address::default(),
        })
    }
}

#[async_trait::async_trait]
impl<C> hopr_api::chain::ChainWriteServiceOperations for TestChainConnector<C>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    type Error = TestConnectorError;

    async fn register_service(
        &self,
        _service_type: hopr_api::chain::ServiceType,
        _metadata: hopr_api::chain::ServiceMetadata,
    ) -> Result<futures::future::BoxFuture<'_, Result<hopr_api::chain::ChainReceipt, Self::Error>>, Self::Error> {
        Err(unsupported_by_test_connector("register_service"))
    }

    async fn update_service(
        &self,
        _service_type: hopr_api::chain::ServiceType,
        _metadata: hopr_api::chain::ServiceMetadata,
    ) -> Result<futures::future::BoxFuture<'_, Result<hopr_api::chain::ChainReceipt, Self::Error>>, Self::Error> {
        Err(unsupported_by_test_connector("update_service"))
    }

    async fn deregister_service(
        &self,
        _service_type: hopr_api::chain::ServiceType,
    ) -> Result<futures::future::BoxFuture<'_, Result<hopr_api::chain::ChainReceipt, Self::Error>>, Self::Error> {
        Err(unsupported_by_test_connector("deregister_service"))
    }
}

fn unsupported_by_test_connector(what: &str) -> TestConnectorError {
    TestConnectorError::from(anyhow::anyhow!("{what} is not supported by TestChainConnector"))
}

// ── ChainReadChannelOperations ────────────────────────────────────────────────

impl<C> hopr_api::chain::ChainReadChannelOperations for TestChainConnector<C>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    type Error = TestConnectorError;

    fn me(&self) -> &Address {
        &self.my_addr
    }

    fn channel_by_id(
        &self,
        channel_id: &ChannelId,
    ) -> Result<Option<hopr_api::types::internal::prelude::ChannelEntry>, Self::Error> {
        Ok(self.channels.get(channel_id).map(|e| *e))
    }

    fn stream_channels<'a>(
        &'a self,
        selector: hopr_api::chain::ChannelSelector,
    ) -> Result<futures::stream::BoxStream<'a, hopr_api::types::internal::prelude::ChannelEntry>, Self::Error> {
        if self.faults.gate_stream(ChainOp::StreamChannels)? == Fault::Hang {
            return Ok(futures::stream::pending().boxed());
        }

        let entries: Vec<_> = self
            .channels
            .iter()
            .filter(|e| selector.satisfies(e.value()))
            .map(|e| *e.value())
            .collect();
        Ok(futures::stream::iter(entries).boxed())
    }
}

// ── ChainWriteChannelOperations ───────────────────────────────────────────────

#[async_trait::async_trait]
impl<C> hopr_api::chain::ChainWriteChannelOperations for TestChainConnector<C>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    type Error = TestConnectorError;

    async fn open_channel<'a>(
        &'a self,
        dst: &'a Address,
        amount: HoprBalance,
    ) -> Result<futures::future::BoxFuture<'a, Result<hopr_api::chain::ChainReceipt, Self::Error>>, Self::Error> {
        self.faults.gate(ChainOp::OpenChannel).await?;

        let channel_id = generate_channel_id(&self.my_addr, dst);
        let tx_req = self.payload_gen()?.fund_channel(*dst, amount)?;
        let receipt = Self::send_tx(
            &self.client,
            tx_req,
            self.chain_id()?,
            &self.chain_key,
            &self.nonce_for(&self.my_addr),
        )
        .await
        .map_err(TestConnectorError::from)?;
        Ok(
            self.track_confirmation(ChainOp::OpenChannel, channel_id, receipt, |channel| {
                channel.status == ChannelStatus::Open
            }),
        )
    }

    async fn fund_channel<'a>(
        &'a self,
        channel_id: &'a ChannelId,
        amount: HoprBalance,
    ) -> Result<futures::future::BoxFuture<'a, Result<hopr_api::chain::ChainReceipt, Self::Error>>, Self::Error> {
        self.faults.gate(ChainOp::FundChannel).await?;

        let channel = self
            .channels
            .get(channel_id)
            .map(|e| *e)
            .ok_or_else(|| anyhow::anyhow!("channel {channel_id} not found"))?;

        let tx_req = self.payload_gen()?.fund_channel(channel.destination, amount)?;
        let funded_to = channel.balance + amount;
        let receipt = Self::send_tx(
            &self.client,
            tx_req,
            self.chain_id()?,
            &self.chain_key,
            &self.nonce_for(&self.my_addr),
        )
        .await
        .map_err(TestConnectorError::from)?;
        Ok(
            self.track_confirmation(ChainOp::FundChannel, *channel_id, receipt, move |channel| {
                channel.balance >= funded_to
            }),
        )
    }

    async fn close_channel<'a>(
        &'a self,
        channel_id: &'a ChannelId,
    ) -> Result<futures::future::BoxFuture<'a, Result<hopr_api::chain::ChainReceipt, Self::Error>>, Self::Error> {
        self.faults.gate(ChainOp::CloseChannel).await?;

        let channel = self
            .channels
            .get(channel_id)
            .map(|e| *e)
            .ok_or_else(|| anyhow::anyhow!("channel {channel_id} not found"))?;

        let tx_req = match channel.status {
            ChannelStatus::Open => self
                .payload_gen()?
                .initiate_outgoing_channel_closure(channel.destination)?,
            ChannelStatus::PendingToClose(_) => self
                .payload_gen()?
                .finalize_outgoing_channel_closure(channel.destination)?,
            ChannelStatus::Closed => return Err(anyhow::anyhow!("channel {channel_id} is already closed").into()),
        };

        let previous_status = channel.status;
        let receipt = Self::send_tx(
            &self.client,
            tx_req,
            self.chain_id()?,
            &self.chain_key,
            &self.nonce_for(&self.my_addr),
        )
        .await
        .map_err(TestConnectorError::from)?;
        // Closure is two steps (Open → PendingToClose → Closed); either way the status this call
        // moved the channel out of must be gone from our own view before we report success.
        Ok(
            self.track_confirmation(ChainOp::CloseChannel, *channel_id, receipt, move |channel| {
                channel.status != previous_status
            }),
        )
    }
}

// ── ChainReadSafeOperations ───────────────────────────────────────────────────

#[async_trait::async_trait]
impl<C> hopr_api::chain::ChainReadSafeOperations for TestChainConnector<C>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    type Error = TestConnectorError;

    async fn safe_allowance<Cur: hopr_api::types::primitive::prelude::Currency, A: Into<Address> + Send>(
        &self,
        safe_address: A,
    ) -> Result<hopr_api::types::primitive::prelude::Balance<Cur>, Self::Error> {
        let address = safe_address.into();
        if Cur::is::<WxHOPR>() {
            Ok(self
                .client
                .query_safe_allowance(&address.into())
                .await?
                .allowance
                .0
                .parse()?)
        } else if Cur::is::<XDai>() {
            Err(anyhow::anyhow!("cannot query allowance on xDai").into())
        } else {
            Err(anyhow::anyhow!("unsupported currency").into())
        }
    }

    async fn safe_info(
        &self,
        selector: hopr_api::chain::SafeSelector,
    ) -> Result<Option<hopr_api::chain::DeployedSafe>, Self::Error> {
        self.faults.gate(ChainOp::SafeInfo).await?;

        let blokli_selector = match selector {
            hopr_api::chain::SafeSelector::Address(a) => BlokliSafeSelector::SafeAddress(a.into()),
            hopr_api::chain::SafeSelector::Deployer(a) => BlokliSafeSelector::ChainKey(a.into()),
            hopr_api::chain::SafeSelector::NodeAddress(a) => BlokliSafeSelector::RegisteredNode(a.into()),
            hopr_api::chain::SafeSelector::Owner(a) => BlokliSafeSelector::Owner(a.into()),
        };

        if let Some(safe) = self.client.query_safe(blokli_selector).await?.first().cloned() {
            Ok(Some(hopr_api::chain::DeployedSafe {
                address: safe.address.parse::<Address>()?,
                owners: safe
                    .owners
                    .into_iter()
                    .map(|a| a.parse::<Address>())
                    .collect::<Result<Vec<_>, _>>()?,
                module: safe.module_address.parse::<Address>()?,
                registered_nodes: safe
                    .registered_nodes
                    .into_iter()
                    .map(|a| a.parse::<Address>())
                    .collect::<Result<Vec<_>, _>>()?,
                deployer: safe.chain_key.parse::<Address>()?,
            }))
        } else {
            Ok(None)
        }
    }

    async fn await_safe_deployment(
        &self,
        selector: hopr_api::chain::SafeSelector,
        _timeout: std::time::Duration,
    ) -> Result<hopr_api::chain::DeployedSafe, Self::Error> {
        self.safe_info(selector)
            .await?
            .ok_or_else(|| TestConnectorError::from(anyhow::anyhow!("safe not found")))
    }

    async fn predict_module_address(
        &self,
        _nonce: u64,
        _owner: &Address,
        _safe_address: &Address,
    ) -> Result<Address, Self::Error> {
        Err(TestConnectorError::from(anyhow::anyhow!(
            "not supported by TestChainConnector"
        )))
    }
}

// ── ChainWriteSafeOperations ──────────────────────────────────────────────────

#[async_trait::async_trait]
impl<C> hopr_api::chain::ChainWriteSafeOperations for TestChainConnector<C>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    type Error = TestConnectorError;

    async fn deploy_safe<'a>(
        &'a self,
        _balance: HoprBalance,
    ) -> Result<futures::future::BoxFuture<'a, Result<hopr_api::chain::ChainReceipt, Self::Error>>, Self::Error> {
        Err(TestConnectorError::from(anyhow::anyhow!(
            "not supported by TestChainConnector"
        )))
    }
}

// ── ChainEvents ───────────────────────────────────────────────────────────────

impl<C> hopr_api::chain::ChainEvents for TestChainConnector<C>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    type Error = TestConnectorError;

    fn subscribe_with_state_sync<I: IntoIterator<Item = hopr_api::chain::StateSyncOptions>>(
        &self,
        _options: I,
    ) -> Result<impl futures::Stream<Item = hopr_api::chain::ChainEvent> + Send + 'static, Self::Error> {
        // async_broadcast::try_broadcast returns TrySendError::Inactive when receiver_count == 0
        // (only InactiveReceiver present), so events fired before the first subscribe() call are
        // silently dropped. Prepend a current-state snapshot to cover any missed transitions.
        let snapshot: Vec<hopr_api::chain::ChainEvent> = self
            .channels
            .iter()
            .filter_map(|e| {
                let ch = *e.value();
                match ch.status {
                    ChannelStatus::Open => Some(hopr_api::chain::ChainEvent::ChannelOpened(ch)),
                    ChannelStatus::PendingToClose(_) => Some(hopr_api::chain::ChainEvent::ChannelClosureInitiated(ch)),
                    ChannelStatus::Closed => None,
                }
            })
            .collect();
        // Withheld kinds are filtered per subscriber: the chain state still
        // changes, only the notification is lost — exactly what an overflowing
        // event broadcast does to a slow consumer.
        // Boxed so the returned stream stays `Unpin` for callers that poll it
        // directly (`next().timeout(..)`), which `Filter` is not.
        let faults = self.faults.clone();
        Ok(futures::stream::iter(snapshot)
            .chain(self.events.1.activate_cloned())
            .filter(move |event| futures::future::ready(!faults.is_withheld(event)))
            .boxed())
    }
}

// ── ChainKeyOperations ────────────────────────────────────────────────────────

impl<C> hopr_api::chain::ChainKeyOperations for TestChainConnector<C>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    type Error = TestConnectorError;
    type Mapper = NoopKeyMapper;

    fn chain_key_to_packet_key(
        &self,
        chain: &Address,
    ) -> Result<Option<hopr_api::types::crypto::prelude::OffchainPublicKey>, Self::Error> {
        Ok(self.accounts.get(chain).map(|e| e.public_key))
    }

    fn packet_key_to_chain_key(
        &self,
        packet: &hopr_api::types::crypto::prelude::OffchainPublicKey,
    ) -> Result<Option<Address>, Self::Error> {
        Ok(self
            .accounts
            .iter()
            .find(|e| &e.value().public_key == packet)
            .map(|e| e.value().chain_addr))
    }

    fn key_id_mapper_ref(&self) -> &Self::Mapper {
        static NOOP: NoopKeyMapper = NoopKeyMapper;
        &NOOP
    }
}

// ── ChainValues ───────────────────────────────────────────────────────────────

#[async_trait::async_trait]
impl<C> hopr_api::chain::ChainValues for TestChainConnector<C>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    type Error = TestConnectorError;

    async fn balance<Cur: hopr_api::types::primitive::prelude::Currency, A: Into<Address> + Send>(
        &self,
        address: A,
    ) -> Result<hopr_api::types::primitive::prelude::Balance<Cur>, Self::Error> {
        self.faults.gate(ChainOp::Balance).await?;

        let address = address.into();
        if Cur::is::<WxHOPR>() {
            Ok(self
                .client
                .query_token_balance(&address.into(), blokli_client::types::Token::WxHOPR)
                .await?
                .balance
                .0
                .parse()?)
        } else if Cur::is::<XDai>() {
            Ok(self
                .client
                .query_native_balance(&address.into())
                .await?
                .balance
                .0
                .parse()?)
        } else {
            Err(anyhow::anyhow!("unsupported currency").into())
        }
    }

    async fn domain_separators(&self) -> Result<hopr_api::chain::DomainSeparators, Self::Error> {
        Ok(self.fetch_parsed_chain_info().await?.domain_separators)
    }

    async fn minimum_incoming_ticket_win_prob(&self) -> Result<WinningProbability, Self::Error> {
        self.faults.gate(ChainOp::WinProb).await?;

        Ok(self.fetch_parsed_chain_info().await?.ticket_win_prob)
    }

    async fn minimum_ticket_price(&self) -> Result<HoprBalance, Self::Error> {
        self.faults.gate(ChainOp::TicketPrice).await?;

        Ok(self.fetch_parsed_chain_info().await?.ticket_price)
    }

    async fn key_binding_fee(&self) -> Result<HoprBalance, Self::Error> {
        let info = self.client.query_chain_info().await?;
        info.key_binding_fee
            .0
            .parse()
            .map_err(|e| TestConnectorError::from(anyhow::anyhow!("invalid key binding fee: {e}")))
    }

    async fn channel_closure_notice_period(&self) -> Result<std::time::Duration, Self::Error> {
        Ok(self.fetch_parsed_chain_info().await?.closure_grace_period)
    }

    async fn chain_info(&self) -> Result<hopr_api::chain::ChainInfo, Self::Error> {
        Ok(self.fetch_parsed_chain_info().await?.chain_info)
    }

    async fn redemption_stats<A: Into<Address> + Send>(
        &self,
        safe_addr: A,
    ) -> Result<hopr_api::chain::RedemptionStats, Self::Error> {
        let safe_addr = safe_addr.into();
        let stats = self
            .client
            .query_redeemed_stats(RedeemedStatsSelector::SafeAddress(safe_addr.into()))
            .await?;
        Ok(hopr_api::chain::RedemptionStats {
            redeemed_count: stats
                .redemption_count
                .0
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid redemption count"))?,
            redeemed_value: stats
                .redeemed_amount
                .0
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid redeemed amount"))?,
        })
    }

    async fn typical_resolution_time(&self) -> Result<std::time::Duration, Self::Error> {
        self.faults.gate(ChainOp::ResolutionTime).await?;

        Ok(std::time::Duration::from_secs(5))
    }
}

// ── ChainReadTicketOperations ─────────────────────────────────────────────────

impl<C> hopr_api::chain::ChainReadTicketOperations for TestChainConnector<C>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    type Error = TestConnectorError;

    fn incoming_ticket_values(&self) -> Result<(WinningProbability, HoprBalance), Self::Error> {
        let win_prob = self
            .ticket_win_prob
            .get()
            .copied()
            .ok_or_else(|| TestConnectorError::from(anyhow::anyhow!("connector not connected")))?;
        let price = self
            .ticket_price
            .get()
            .cloned()
            .ok_or_else(|| TestConnectorError::from(anyhow::anyhow!("connector not connected")))?;
        Ok((win_prob, price))
    }
}

// ── ChainWriteTicketOperations ────────────────────────────────────────────────

#[async_trait::async_trait]
impl<C> hopr_api::chain::ChainWriteTicketOperations for TestChainConnector<C>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    type Error = TestConnectorError;

    async fn redeem_ticket<'a>(
        &'a self,
        ticket: RedeemableTicket,
    ) -> Result<
        futures::future::BoxFuture<
            'a,
            Result<(VerifiedTicket, hopr_api::chain::ChainReceipt), hopr_api::chain::TicketRedeemError<Self::Error>>,
        >,
        hopr_api::chain::TicketRedeemError<Self::Error>,
    > {
        let verified_ticket = ticket.ticket;
        let tx_req = self
            .payload_gen()
            .map_err(|e| {
                hopr_api::chain::TicketRedeemError::ProcessingError(verified_ticket, TestConnectorError::from(e))
            })?
            .redeem_ticket(ticket)
            .map_err(|e| {
                hopr_api::chain::TicketRedeemError::ProcessingError(
                    verified_ticket,
                    TestConnectorError::from(anyhow::anyhow!("{e}")),
                )
            })?;

        let chain_id = self.chain_id().map_err(|e| {
            hopr_api::chain::TicketRedeemError::ProcessingError(verified_ticket, TestConnectorError::from(e))
        })?;
        let chain_key = self.chain_key.clone();
        let nonce = self.nonce_for(&self.my_addr);
        let client = self.client.clone();

        Ok(Box::pin(async move {
            let receipt = Self::send_tx(&client, tx_req, chain_id, &chain_key, &nonce)
                .await
                .map_err(|e| {
                    hopr_api::chain::TicketRedeemError::ProcessingError(verified_ticket, TestConnectorError::from(e))
                })?;
            Ok((verified_ticket, receipt))
        }))
    }
}

// ── ComponentStatusReporter ───────────────────────────────────────────────────

impl<C> hopr_api::node::ComponentStatusReporter for TestChainConnector<C>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    fn component_status(&self) -> hopr_api::node::ComponentStatus {
        hopr_api::node::ComponentStatus::Ready
    }
}

// ── PacketTransport ───────────────────────────────────────────────────────────

impl<C> hopr_api::node::PacketTransport for TestChainConnector<C>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    fn packet_payload_size() -> usize {
        1036
    }
}

// ── Factory function ──────────────────────────────────────────────────────────

/// Creates and connects a [`TestChainConnector`] backed by the given Blokli client.
///
/// Equivalent to `create_trustful_hopr_blokli_connector` but without the
/// `hopr-chain-connector` git dependency.
pub async fn create_test_blokli_connector<C>(
    chain_key: &hopr_api::types::crypto::prelude::ChainKeypair,
    client: C,
    module_address: Address,
) -> anyhow::Result<TestChainConnector<C>>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    let my_addr = chain_key.public().to_address();
    let mut connector = TestChainConnector::new(client, my_addr, chain_key.clone(), module_address);
    connector.connect().await?;
    Ok(connector)
}

/// Registers `node_address` in its pre-created safe.
///
/// After [`create_test_blokli_connector`] connects a node, call this so that
/// `safe_info(NodeAddress(me))` returns the safe — a prerequisite for strategies
/// that read the safe balance (e.g. `auto_funding`).
pub async fn register_test_safe<C>(connector: &TestChainConnector<C>, node_address: Address) -> anyhow::Result<()>
where
    C: BlokliQueryClient + BlokliSubscriptionClient + BlokliTransactionClient + Clone + Send + Sync + 'static,
{
    use hopr_api::chain::{AccountSelector, ChainReadAccountOperations, ChainWriteAccountOperations};

    let account = connector
        .stream_accounts(AccountSelector::default().with_chain_key(node_address))
        .map_err(anyhow::Error::from)?
        .next()
        .await
        .ok_or_else(|| anyhow::anyhow!("account not found for {node_address}"))?;

    let safe_address = account
        .safe_address
        .ok_or_else(|| anyhow::anyhow!("no safe address for node {node_address}"))?;

    connector
        .register_safe(&safe_address)
        .await
        .map_err(|e| anyhow::anyhow!("register_safe submission failed: {e}"))?
        .await
        .map_err(|e| anyhow::anyhow!("register_safe confirmation failed: {e}"))?;

    Ok(())
}
