// The plain network- and session-related types now live in `hopr-types`; re-exported here for
// backward compatibility. Runtime-specific behavior (DNS resolution) stays in this crate via the
// [`IpOrHostExt`] extension trait below.
pub use hopr_types::network_types::{IpOrHost, IpProtocol, SealedHost, ServiceId, SessionId, SessionTarget};

/// Runtime-specific extensions for [`IpOrHost`].
#[cfg(feature = "network-types-runtime-tokio")]
pub trait IpOrHostExt {
    /// Tries to resolve the DNS name and returns all IP addresses found.
    /// If this is already an IP address and port, it will simply return it.
    ///
    /// Uses `tokio` resolver.
    fn resolve_tokio(self) -> impl std::future::Future<Output = std::io::Result<Vec<std::net::SocketAddr>>> + Send;
}

#[cfg(feature = "network-types-runtime-tokio")]
impl IpOrHostExt for IpOrHost {
    async fn resolve_tokio(self) -> std::io::Result<Vec<std::net::SocketAddr>> {
        match self {
            IpOrHost::Dns(name, port) => {
                static RESOLVER: tokio::sync::OnceCell<hickory_resolver::TokioResolver> =
                    tokio::sync::OnceCell::const_new();

                let resolver = RESOLVER
                    .get_or_try_init(|| async {
                        hickory_resolver::Resolver::builder_tokio()
                            .map_err(std::io::Error::other)?
                            .build()
                            .map_err(std::io::Error::other)
                    })
                    .await?;

                let lookup = resolver.lookup_ip(&name).await.map_err(std::io::Error::other)?;
                Ok(lookup.iter().map(|ip| std::net::SocketAddr::new(ip, port)).collect())
            }
            IpOrHost::Ip(addr) => Ok(vec![addr]),
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "network-types", feature = "runtime-tokio"))]
    use {super::*, anyhow::anyhow, std::net::SocketAddr};

    #[cfg(all(feature = "network-types", feature = "runtime-tokio"))]
    #[tokio::test]
    async fn ip_or_host_must_resolve_ip_address() -> anyhow::Result<()> {
        let actual = IpOrHost::Ip("127.0.0.1:1000".parse()?).resolve_tokio().await?;

        let actual = actual.first().ok_or(anyhow!("must resolve"))?;

        let expected: SocketAddr = "127.0.0.1:1000".parse()?;

        assert_eq!(*actual, expected);
        Ok(())
    }
}
