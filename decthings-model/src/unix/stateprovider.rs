use std::collections::HashSet;

use crate::*;
use futures::future::BoxFuture;

struct StateProviderImpl<'a> {
    command_id: &'a str,
    sender: super::host_protocol::Sender,
    provided: HashSet<String>,
}

impl<'a> StateProvider for StateProviderImpl<'a> {
    fn provide_all<'b>(
        &'b mut self,
        data: &'b [(impl AsRef<str> + Send + Sync + 'b, bytes::Bytes)],
    ) -> BoxFuture<'b, ()> {
        Box::pin(async move {
            for (key, _) in data {
                if !self.provided.insert(key.as_ref().to_owned()) {
                    panic!(
                        r#"StateProvider: State key "{}" was provided multiple times."#,
                        key.as_ref()
                    );
                }
            }
            if self.provided.len() as u32 + data.len() as u32 > 100 {
                panic!("StateProvider: Cannot provide more than 100 keys.");
            }
            if data.iter().any(|(_, data)| data.len() > 1024usize.pow(3)) {
                panic!("StateProvider: Cannot provide more than 1 gigabyte for a single key. Split it into multiple keys.");
            }

            // Output in batches of at most 1 GiB
            let mut i = 0;
            while i < data.len() {
                let mut names = vec![data[i].0.as_ref()];
                let mut to_send = vec![data[i].1.clone()];
                i += 1;
                let mut total_length = to_send[0].len();
                while i < data.len() && total_length + data[i].1.len() < 1024usize.pow(3) {
                    total_length += data[i].1.len();
                    names.push(data[i].0.as_ref());
                    to_send.push(data[i].1.clone());
                    i += 1;
                }
                self.sender
                    .send_event(
                        super::host_protocol::EventMessage::ProvideStateData {
                            command_id: self.command_id,
                            names: &names,
                        },
                        to_send,
                    )
                    .await;
            }
        })
    }
}

pub(super) fn create_state_provider(
    command_id: &str,
    sender: super::host_protocol::Sender,
) -> impl StateProvider + '_ {
    StateProviderImpl {
        command_id,
        sender,
        provided: HashSet::new(),
    }
}
