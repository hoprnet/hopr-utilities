mod emulator;

use blokli_client::BlokliTestStateMutator;
pub use blokli_client::{BlokliTestClient, BlokliTestState, exports::Entry};
pub use emulator::{ChainMutator, FullStateEmulator, StaticState};
pub use hopr_api::chain::ChainInfo;
use hopr_api::{
    chain::{DeployedSafe, ServiceEntry, ServiceRegistryConfig, ServiceType, ServiceTypeConfig},
    types::{
        chain::{ParsedHoprChainAction, contract_addresses_for_network},
        crypto::{
            prelude::{Keypair, OffchainKeypair},
            types::Hash,
        },
        internal::prelude::*,
        primitive::prelude::*,
    },
};

/// Allows easily building the [`BlokliTestState`] using the HOPR native types.
#[derive(Clone)]
pub struct BlokliTestStateBuilder(BlokliTestState);

const GVPN_EXIT_REGISTRATION_BURN: &str = "1000000000000000000000 wei wxHOPR";
const GVPN_EXIT_UPDATE_BURN: &str = "100000000000000000000 wei wxHOPR";
const SERVICE_TYPE_REGISTRATION_FEE: &str = "1000000000000000000 wei wxHOPR";
const DEFAULT_ALLOWANCE: u128 = 10_000_000_000_000_u128;

impl Default for BlokliTestStateBuilder {
    fn default() -> Self {
        let (_, addresses) = contract_addresses_for_network("anvil-localhost").expect("network name not found");

        Self(BlokliTestState::default())
            .with_hopr_network_chain_info("anvil-localhost")
            .with_service_types([(
                ServiceType::GVPN_EXIT,
                ServiceTypeConfig {
                    owner: Some(
                        "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266"
                            .parse()
                            .expect("default anvil deployer address must be valid"),
                    ),
                    requirement: None,
                    registration_burn: GVPN_EXIT_REGISTRATION_BURN
                        .parse()
                        .expect("default registration burn must be valid"),
                    update_burn: GVPN_EXIT_UPDATE_BURN
                        .parse()
                        .expect("default update burn must be valid"),
                },
            )])
            .with_service_registry_config(ServiceRegistryConfig {
                type_registration_fee: SERVICE_TYPE_REGISTRATION_FEE
                    .parse()
                    .expect("default service type registration fee must be valid"),
                node_safe_registry: Address::new(addresses.node_safe_registry.as_slice()),
            })
    }
}

/// Converts a timestamp into the unsigned Unix seconds represented by the Blokli API.
///
/// Panics for a timestamp before the Unix epoch.
fn unix_seconds(time: std::time::SystemTime) -> blokli_client::api::types::Uint64 {
    blokli_client::api::types::Uint64(
        time.duration_since(std::time::UNIX_EPOCH)
            .expect("timestamp must not precede the Unix epoch")
            .as_secs()
            .to_string(),
    )
}

impl From<BlokliTestState> for BlokliTestStateBuilder {
    fn from(state: BlokliTestState) -> Self {
        Self(state)
    }
}

impl BlokliTestStateBuilder {
    /// Appends the initial [`ChannelEntries`](ChannelEntry) in the state.
    ///
    /// The function will panic if any channels refer to accounts which were not previously
    /// present in the state or added via a [`BlokliTestStateBuilder::with_accounts`] call.
    #[must_use]
    pub fn with_channels<I: IntoIterator<Item = ChannelEntry>>(mut self, channels: I) -> Self {
        self.0.channels.extend(channels.into_iter().map(|channel| {
            (
                const_hex::encode(channel.get_id()),
                blokli_client::api::types::Channel {
                    balance: blokli_client::api::types::TokenValueString(channel.balance.to_string()),
                    closure_time: if let ChannelStatus::PendingToClose(time) = channel.status {
                        Some(blokli_client::api::types::DateTime(
                            hopr_api::chain::DateTime::from(time).to_rfc3339(),
                        ))
                    } else {
                        None
                    },
                    concrete_channel_id: const_hex::encode(channel.get_id()),
                    source: self
                        .0
                        .accounts
                        .values()
                        .find(|a| a.chain_key == const_hex::encode(channel.source))
                        .map(|a| a.keyid)
                        .unwrap_or_else(|| panic!("missing dst account {}", channel.source)),
                    epoch: channel.channel_epoch as i32,
                    destination: self
                        .0
                        .accounts
                        .values()
                        .find(|a| a.chain_key == const_hex::encode(channel.destination))
                        .map(|a| a.keyid)
                        .unwrap_or_else(|| panic!("missing dst account {}", channel.destination)),
                    status: match channel.status {
                        ChannelStatus::Closed => blokli_client::api::types::ChannelStatus::Closed,
                        ChannelStatus::Open => blokli_client::api::types::ChannelStatus::Open,
                        ChannelStatus::PendingToClose(_) => blokli_client::api::types::ChannelStatus::PendingToClose,
                    },
                    ticket_index: blokli_client::api::types::Uint64(channel.ticket_index.to_string()),
                },
            )
        }));
        self
    }

    /// Appends the initial [`AccountEntries`](AccountEntry) in the state.
    #[must_use]
    pub fn with_accounts<I: IntoIterator<Item = (AccountEntry, HoprBalance, XDaiBalance)>>(
        mut self,
        accounts: I,
    ) -> Self {
        for (account, hopr_balance, native_balance) in accounts {
            match self.0.accounts.entry(account.key_id.into()) {
                Entry::Occupied(_) => panic!("duplicate key id for account {}", account.chain_addr),
                Entry::Vacant(v) => {
                    v.insert(blokli_client::api::types::Account {
                        chain_key: const_hex::encode(account.chain_addr),
                        keyid: u32::from(account.key_id) as i32,
                        multi_addresses: account.get_multiaddrs().iter().map(|a| a.to_string()).collect(),
                        packet_key: const_hex::encode(account.public_key),
                        safe_address: account.safe_address.map(const_hex::encode),
                    });
                    if let Some(safe_addr) = account.safe_address.as_ref().map(const_hex::encode) {
                        self.0.deployed_safes.insert(
                            safe_addr.clone(),
                            blokli_client::api::types::Safe {
                                address: safe_addr,
                                chain_key: const_hex::encode(account.chain_addr),
                                owners: [const_hex::encode(account.chain_addr)].to_vec(),
                                module_address: const_hex::encode(
                                    &Hash::create(&[account.chain_addr.as_ref()]).as_ref()[0..Address::SIZE],
                                ),
                                registered_nodes: vec![],
                                threshold: Some("1".to_string()),
                            },
                        );
                    }
                    self.0.token_balances.insert(
                        const_hex::encode(account.chain_addr),
                        blokli_client::api::types::HoprBalance {
                            __typename: "HoprBalance".to_string(),
                            balance: blokli_client::api::types::TokenValueString(HoprBalance::zero().to_string()),
                        },
                    );
                    self.0.native_balances.insert(
                        const_hex::encode(account.chain_addr),
                        blokli_client::api::types::NativeBalance {
                            __typename: "NativeBalance".to_string(),
                            balance: blokli_client::api::types::TokenValueString(native_balance.to_string()),
                        },
                    );
                    if let Some(addr) = account.safe_address.as_ref().map(const_hex::encode) {
                        self.0.token_balances.insert(
                            addr.clone(),
                            blokli_client::api::types::HoprBalance {
                                __typename: "HoprBalance".to_string(),
                                balance: blokli_client::api::types::TokenValueString(hopr_balance.to_string()),
                            },
                        );
                        self.0.native_balances.insert(
                            addr.clone(),
                            blokli_client::api::types::NativeBalance {
                                __typename: "NativeBalance".to_string(),
                                balance: blokli_client::api::types::TokenValueString(XDaiBalance::zero().to_string()),
                            },
                        );
                        self.0.safe_allowances.insert(
                            addr.clone(),
                            blokli_client::api::types::SafeHoprAllowance {
                                __typename: "SafeHoprAllowance".to_string(),
                                allowance: blokli_client::api::types::TokenValueString(
                                    HoprBalance::new_base(DEFAULT_ALLOWANCE).to_string(),
                                ),
                            },
                        );
                    }
                }
            }
        }
        self
    }

    /// Appends the initial [`DeployedSafes`](DeployedSafe) to the state.
    pub fn with_deployed_safes<I: IntoIterator<Item = DeployedSafe>>(mut self, safes: I) -> Self {
        self.0.deployed_safes.extend(safes.into_iter().map(|safe| {
            (
                const_hex::encode(safe.address),
                blokli_client::api::types::Safe {
                    address: const_hex::encode(safe.address),
                    chain_key: const_hex::encode(safe.deployer),
                    owners: safe.owners.into_iter().map(const_hex::encode).collect(),
                    module_address: const_hex::encode(safe.module),
                    registered_nodes: safe.registered_nodes.into_iter().map(const_hex::encode).collect(),
                    threshold: Some("1".to_string()),
                },
            )
        }));
        self
    }

    /// Appends the initial [`ServiceEntries`](ServiceEntry) of the on-chain service registry to the state.
    ///
    /// The service type of each entry is rendered the way Blokli renders it: the ASCII name of the type, or
    /// `0x`-prefixed hex for a type that does not follow that convention.
    #[must_use]
    pub fn with_services<I: IntoIterator<Item = ServiceEntry>>(mut self, services: I) -> Self {
        for service in services {
            let service_type = service.service_type.to_string();
            match self
                .0
                .services
                .entry(BlokliTestState::service_entry_key(&service_type, &service.node.into()))
            {
                Entry::Occupied(_) => panic!(
                    "duplicate service entry for service type {service_type} of node {}",
                    service.node
                ),
                Entry::Vacant(v) => {
                    v.insert(blokli_client::api::types::ServiceEntry {
                        service_type,
                        node: const_hex::encode(service.node),
                        safe: const_hex::encode(service.safe),
                        metadata: format!("0x{}", const_hex::encode(&service.metadata)),
                        registered_at: unix_seconds(service.registered_at),
                        updated_at: unix_seconds(service.updated_at),
                    });
                }
            }
        }
        self
    }

    /// Appends the initial [`ServiceTypeConfigs`](ServiceTypeConfig) of the on-chain service registry to the state.
    #[must_use]
    pub fn with_service_types<I: IntoIterator<Item = (ServiceType, ServiceTypeConfig)>>(
        mut self,
        service_types: I,
    ) -> Self {
        for (service_type, config) in service_types {
            let service_type = service_type.to_string();
            match self.0.service_types.entry(service_type.clone()) {
                Entry::Occupied(_) => panic!("duplicate service type {service_type}"),
                Entry::Vacant(v) => {
                    v.insert(blokli_client::api::types::ServiceTypeInfo {
                        service_type,
                        owner: config.owner.map(const_hex::encode),
                        requirement: config.requirement.map(const_hex::encode),
                        registration_burn: config.registration_burn.to_string(),
                        update_burn: config.update_burn.to_string(),
                    });
                }
            }
        }
        self
    }

    /// Sets the initial registry-wide service configuration.
    #[must_use]
    pub fn with_service_registry_config(mut self, config: ServiceRegistryConfig) -> Self {
        self.0.service_registry_config = blokli_client::api::types::ServiceRegistryConfig {
            type_registration_fee: config.type_registration_fee.to_string(),
            node_safe_registry: const_hex::encode(config.node_safe_registry),
        };
        self
    }

    /// Generates [`AccountEntries`](AccountEntry) for the given addresses.
    ///
    /// The off-chain keys and safe addresses are chosen deterministically using a
    /// pseudorandom function of each given address.
    #[must_use]
    pub fn with_generated_accounts(
        self,
        addresses: &[&Address],
        public: bool,
        native: XDaiBalance,
        token: HoprBalance,
    ) -> Self {
        let max_id = self.0.accounts.keys().max().copied().unwrap_or(0);
        self.with_accounts(addresses.iter().enumerate().map(|(index, &chain_addr)| {
            let pseudorandom_data = Hash::create(&[chain_addr.as_ref()]);
            let ok = OffchainKeypair::from_secret(pseudorandom_data.as_ref())
                .expect("offchain keypair creation cannot fail");
            let safe_addr = pseudorandom_data.hash();
            (
                AccountEntry {
                    public_key: *ok.public(),
                    chain_addr: *chain_addr,
                    entry_type: if public {
                        AccountType::Announced(vec![
                            format!("/ip4/1.2.3.4/udp/{}/p2p/{}", 10000 + index, ok.public().to_peerid_str())
                                .parse()
                                .unwrap(),
                        ])
                    } else {
                        AccountType::NotAnnounced
                    },
                    safe_address: Some(Address::new(&safe_addr.as_ref()[0..Address::SIZE])),
                    key_id: KeyIdent::from(max_id + index as u32),
                },
                token,
                native,
            )
        }))
    }

    #[must_use]
    pub fn with_balances<C: Currency>(mut self, balances: impl IntoIterator<Item = (Address, Balance<C>)>) -> Self {
        if C::is::<XDai>() {
            self.0
                .native_balances
                .extend(balances.into_iter().map(|(addr, balance)| {
                    (
                        const_hex::encode(addr),
                        blokli_client::api::types::NativeBalance {
                            __typename: "NativeBalance".into(),
                            balance: blokli_client::api::types::TokenValueString(balance.to_string()),
                        },
                    )
                }))
        } else if C::is::<WxHOPR>() {
            self.0
                .token_balances
                .extend(balances.into_iter().map(|(addr, balance)| {
                    (
                        const_hex::encode(addr),
                        blokli_client::api::types::HoprBalance {
                            __typename: "HoprBalance".into(),
                            balance: blokli_client::api::types::TokenValueString(balance.to_string()),
                        },
                    )
                }))
        } else {
            panic!("unsupported currency");
        }

        self
    }

    /// Sets the initial Safe allowances for the given Safe addresses.
    #[must_use]
    pub fn with_safe_allowances<I: IntoIterator<Item = (Address, HoprBalance)>>(mut self, balances: I) -> Self {
        self.0
            .safe_allowances
            .extend(balances.into_iter().map(|(addr, allowance)| {
                (
                    const_hex::encode(addr),
                    blokli_client::api::types::SafeHoprAllowance {
                        __typename: "SafeAllowance".into(),
                        allowance: blokli_client::api::types::TokenValueString(allowance.to_string()),
                    },
                )
            }));
        self
    }

    /// Sets [`ChainInfo`] to the state. If not set, the default values are used.
    #[must_use]
    pub fn with_chain_info(mut self, info: ChainInfo) -> Self {
        self.0.chain_info.chain_id = info.chain_id as i32;
        self.0.chain_info.network = info.hopr_network_name;
        self.0.chain_info.contract_addresses = blokli_client::api::types::ContractAddressMap(
            serde_json::to_string(&info.contract_addresses).expect("failed to serialize contract addresses"),
        );
        self
    }

    /// Set chain info based on the known HOPR network name.
    ///
    /// Panics if such a network deployment does not exist.
    #[must_use]
    pub fn with_hopr_network_chain_info(mut self, name: &str) -> Self {
        let (chain_id, addrs) = contract_addresses_for_network(name).expect("network name not found");
        self.0.chain_info.network = name.to_string();
        self.0.chain_info.contract_addresses = blokli_client::api::types::ContractAddressMap(
            serde_json::to_string(&addrs).expect("failed to serialize contract addresses"),
        );
        self.0.chain_info.chain_id = chain_id as i32;
        self
    }

    /// Sets the ticket price. If not set, the default value is used.
    #[must_use]
    pub fn with_ticket_price(mut self, price: HoprBalance) -> Self {
        self.0.chain_info.ticket_price = blokli_client::api::types::TokenValueString(price.to_string());
        self
    }

    /// Sets the minimum winning probability. If not set, the default value is used.
    #[must_use]
    pub fn with_minimum_win_prob(mut self, prob: WinningProbability) -> Self {
        self.0.chain_info.min_ticket_winning_probability = prob.as_f64();
        self
    }

    /// Sets the channel closure grace period. If not set, the default value is used.
    #[must_use]
    pub fn with_closure_grace_period(mut self, grace_period: std::time::Duration) -> Self {
        self.0.chain_info.channel_closure_grace_period =
            blokli_client::api::types::Uint64(grace_period.as_secs().to_string());
        self
    }

    /// Builds the state.
    #[must_use]
    pub fn build(self) -> BlokliTestState {
        self.0
    }

    /// Builds the state and returns a [`BlokliTestClient`] that cannot mutate the state.
    ///
    /// This is useful for simple static tests that only read data from the Chain.
    #[must_use]
    pub fn build_static_client(self) -> BlokliTestClient<StaticState> {
        BlokliTestClient::new(self.0, StaticState)
    }

    /// Builds the state and returns a [`BlokliTestClient`] that can also mutate the state.
    ///
    /// This is useful for more advanced dynamic tests which involve on-chain interactions.
    ///
    /// Because the underlying client performs all transactions via Safe, a `module_address` must be
    /// given.
    ///
    /// If the returned client is cloned, the state will be shared amongst them, and each
    /// client will see the changes made by others.
    #[must_use]
    pub fn build_dynamic_client(self, module_address: Address) -> BlokliTestClient<FullStateEmulator> {
        self.build_dynamic_client_with_mutator(FullStateEmulator(module_address, None))
    }

    /// Builds the state and returns a [`BlokliTestClient`] that can also mutate the state.
    ///
    /// This is useful for more advanced dynamic tests which involve on-chain interactions.
    ///
    /// See [`BlokliTestStateBuilder::build_dynamic_client`] for simplified usage.
    #[must_use]
    pub fn build_dynamic_client_with_mutator<M: BlokliTestStateMutator>(self, mutator: M) -> BlokliTestClient<M> {
        BlokliTestClient::new(self.0, mutator)
    }

    /// Similar to [`BlokliTestStateBuilder::build_dynamic_client`], but also creates
    /// a stream of [`ParsedHoprChainAction`] to allow interception of all on-chain actions
    /// sent by the client.
    pub fn build_dynamic_client_with_tx_interceptor(
        self,
        module_address: Address,
    ) -> (
        BlokliTestClient<FullStateEmulator>,
        impl futures::Stream<Item = ParsedHoprChainAction>,
    ) {
        let (sender, receiver) = futures::channel::mpsc::unbounded();
        let client = BlokliTestClient::new(self.0, FullStateEmulator(module_address, Some(sender)));
        (client, receiver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_matches_the_deployed_gvpn_exit_type() {
        let state = BlokliTestStateBuilder::default().build();
        let service_type = state
            .get_service_type(&ServiceType::GVPN_EXIT.as_encoded())
            .expect("gvpn:exit must be registered by default");

        assert_eq!(service_type.service_type, "gvpn:exit");
        assert_eq!(
            service_type.owner.as_deref(),
            Some("f39fd6e51aad88f6f4ce6ab8827279cfffb92266")
        );
        assert_eq!(service_type.requirement, None);
        assert_eq!(service_type.registration_burn, "1000 wxHOPR");
        assert_eq!(service_type.update_burn, "100 wxHOPR");
        assert_eq!(state.service_registry_config.type_registration_fee, "1 wxHOPR");
        assert!(state.services.is_empty());
    }
}
