use cdrs_tokio::cluster::session::{TcpSessionBuilder, SessionBuilder, Session};
use cdrs_tokio::cluster::{NodeTcpConfigBuilder, TcpConnectionManager};
use cdrs_tokio::load_balancing::RoundRobinLoadBalancingStrategy;
use cdrs_tokio::{query::*, query_values};
use cdrs_tokio::transport::TransportTcp;
use cdrs_tokio::{IntoCdrsValue, TryFromRow};

#[derive(Clone, Debug, IntoCdrsValue, TryFromRow, PartialEq)]
pub struct DatabasePokerHand {
    pub cards_id: String,
}

impl DatabasePokerHand {
    fn into_query_values(self) -> QueryValues {
        // **IMPORTANT NOTE:** query values should be WITHOUT NAMES
        // https://github.com/apache/cassandra/blob/trunk/doc/native_protocol_v4.spec#L413
        query_values!(self.cards_id)
    }
}

pub async fn create_session() -> Session<TransportTcp, TcpConnectionManager, RoundRobinLoadBalancingStrategy<TransportTcp, TcpConnectionManager>> {
    let cluster_config = NodeTcpConfigBuilder::new()
        .with_contact_point("127.0.0.1:9042".into())
        .build()
        .await
        .unwrap();
    return TcpSessionBuilder::new(RoundRobinLoadBalancingStrategy::new(), cluster_config)
        .build()
        .await
        .unwrap();
}

pub async fn insert_batch(
    session: &Session<TransportTcp, TcpConnectionManager, RoundRobinLoadBalancingStrategy<TransportTcp, TcpConnectionManager>>,
    hands: Vec<DatabasePokerHand>,
    table: &String,
) {
    let mut batch = BatchQueryBuilder::new();
    for hand in hands {
        let query = format!("INSERT INTO poker_hands.{} (cards_id) VALUES (?)", table);
        batch = batch.add_query(query, hand.into_query_values());
    }

    let batch_query = batch.build().expect("Batch builder");
    session.batch(batch_query).await.expect("Batch query error");
}
