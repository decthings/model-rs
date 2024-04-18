pub use lazy_static::lazy_static;
pub use pollster;

#[allow(clippy::missing_safety_doc)]
pub mod wasm_host_functions {
    use futures::future::BoxFuture;
    use std::collections::{HashMap, HashSet};

    #[no_mangle]
    extern "C" fn dt_malloc(amount: u64) -> u64 {
        let buf: Vec<u8> = Vec::with_capacity(amount.try_into().unwrap());
        let ptr = buf.as_ptr();
        std::mem::forget(buf);
        ptr as u64
    }

    extern "C" {
        pub fn dt_callback(ptr: u64);
        fn dt_provide_state_data(amount: u32, ptr: u64);
        fn dt_request_data(dataset: u32, start_index: u32, amount: u32) -> u64;
        fn dt_shuffle_data(amount: u32, ptr: u64);
        fn dt_training_progress(progress: f32);
        fn dt_training_metrics(amount: u32, ptr: u64);
    }

    #[derive(Debug, PartialEq)]
    pub struct Param {
        pub name: String,
        pub dataset_id: u32,
        pub amount: u32,
        pub total_byte_size: u64,
    }

    pub unsafe fn read_params(
        ptr: u64,
    ) -> HashMap<String, impl crate::DataLoaderBinary + crate::StateLoader> {
        let num_params = u32::from_be_bytes(*(ptr as *const [u8; 4])) as usize;
        let mut pos = ptr + 4;

        let mut res = HashMap::with_capacity(num_params);
        for _ in 0..num_params {
            let dataset_id = u32::from_be_bytes(*(pos as *const [u8; 4]));
            pos += 4;
            let amount = u32::from_be_bytes(*(pos as *const [u8; 4]));
            pos += 4;
            let total_byte_size = u64::from_be_bytes(*(pos as *const [u8; 8]));
            pos += 8;
            let name_len = u64::from_be_bytes(*(pos as *const [u8; 8])) as usize;
            pos += 8;
            let name_ptr = u64::from_be_bytes(*(pos as *const [u8; 8]));
            pos += 8;
            res.insert(
                String::from_utf8_unchecked(Vec::from_raw_parts(
                    name_ptr as *mut u8,
                    name_len,
                    name_len,
                )),
                DataLoaderImpl {
                    dataset_id,
                    size: amount,
                    total_byte_size,
                    position: 0,
                },
            );
        }
        res
    }

    pub fn request_data(dataset: u32, start_index: u32, amount: u32) -> Vec<Vec<u8>> {
        unsafe {
            let res_ptr = dt_request_data(dataset, start_index, amount);

            let num_segments = u32::from_be_bytes(*(res_ptr as *const [u8; 4])) as usize;
            let mut pos = res_ptr + 4;

            let mut res = Vec::with_capacity(num_segments);
            for _ in 0..num_segments {
                let len = u64::from_be_bytes(*(pos as *const [u8; 8])) as usize;
                pos += 8;
                let data_ptr = u64::from_be_bytes(*(pos as *const [u8; 8]));
                pos += 8;
                res.push(Vec::from_raw_parts(data_ptr as *mut u8, len, len));
            }

            drop(Vec::from_raw_parts(
                res_ptr as *mut u8,
                pos as usize,
                pos as usize,
            ));

            res
        }
    }

    struct DataLoaderImpl {
        dataset_id: u32,
        size: u32,
        total_byte_size: u64,
        position: u32,
    }

    impl crate::DataLoaderBinary for DataLoaderImpl {
        fn total_byte_size(&self) -> u64 {
            self.total_byte_size
        }

        fn shuffle_in_group<'b>(&'b self, others: &'b [&'b Self]) -> BoxFuture<'b, ()> {
            let mut buf = Vec::with_capacity(others.len() * 4);
            for other in others {
                buf.extend_from_slice(&other.dataset_id.to_be_bytes());
            }
            unsafe { dt_shuffle_data(others.len() as u32, buf.as_ptr() as u64) };
            Box::pin(async {})
        }

        fn size(&self) -> u32 {
            self.size
        }

        fn position(&self) -> u32 {
            self.position
        }

        fn set_position(&mut self, position: u32) {
            if position >= self.size {
                panic!("DataLoader: Cannot set the the position to a value greater than or equal to the data size. The data size was {}, and position {} was attempted to be set.", self.size, position);
            }
            self.position = position;
        }

        fn remaining(&self) -> u32 {
            self.size - self.position
        }

        fn next(&mut self, mut amount: u32) -> BoxFuture<'_, Vec<bytes::Bytes>> {
            Box::pin(async move {
                amount = amount.min(crate::DataLoaderBinary::remaining(self));

                if amount == 0 {
                    return vec![];
                }

                let prev_position = crate::DataLoaderBinary::position(self);
                self.position += amount;

                request_data(self.dataset_id, prev_position, amount)
                    .into_iter()
                    .map(Into::into)
                    .collect()
            })
        }
    }

    impl crate::StateLoader for DataLoaderImpl {
        fn byte_size(&self) -> u64 {
            crate::DataLoaderBinary::total_byte_size(self)
        }

        fn read(&mut self) -> BoxFuture<'_, bytes::Bytes> {
            Box::pin(async {
                crate::DataLoaderBinary::set_position(self, 0);
                crate::DataLoaderBinary::next(self, 1).await.remove(0)
            })
        }
    }

    pub unsafe fn evaluate_callback(result: Result<Vec<crate::ParameterBinary>, Option<String>>) {
        match result {
            Ok(outputs) => {
                let mut buf = Vec::with_capacity(
                    5 + outputs
                        .iter()
                        .map(|x| 20 + x.data.len() * 16)
                        .sum::<usize>(),
                );
                buf.push(0);
                buf.extend_from_slice(&(outputs.len() as u32).to_be_bytes());
                for param in &outputs {
                    buf.extend_from_slice(&(param.name.len() as u64).to_be_bytes());
                    buf.extend_from_slice(&(param.name.as_ptr() as u64).to_be_bytes());
                    buf.extend_from_slice(&(param.data.len() as u32).to_be_bytes());
                    for data2 in &param.data {
                        buf.extend_from_slice(&(data2.len() as u64).to_be_bytes());
                        buf.extend_from_slice(&(data2.as_ptr() as u64).to_be_bytes());
                    }
                }
                dt_callback(buf.as_ptr() as u64);
                drop(outputs);
            }
            Err(Some(s)) => {
                let mut buf = Vec::with_capacity(9 + s.len());
                buf.push(1);
                buf.extend_from_slice(&(s.len() as u64).to_be_bytes());
                buf.extend_from_slice(&(s.as_ptr() as u64).to_be_bytes());
            }
            Err(None) => {
                let buf = [2];
                dt_callback(buf.as_ptr() as u64);
            }
        }
    }

    pub unsafe fn training_metrics<S: AsRef<str>, D: AsRef<[u8]>>(metrics: &[(S, D)]) {
        let mut buf = Vec::with_capacity(metrics.len() * 32);
        for (name, data) in metrics {
            buf.extend_from_slice(&(name.as_ref().len() as u64).to_be_bytes());
            buf.extend_from_slice(&(name.as_ref().as_ptr() as u64).to_be_bytes());
            buf.extend_from_slice(&(data.as_ref().as_ref().len() as u64).to_be_bytes());
            buf.extend_from_slice(&(data.as_ref().as_ptr() as u64).to_be_bytes());
        }
        dt_training_metrics(metrics.len() as u32, buf.as_ptr() as u64);
    }

    struct TrainTrackerImpl;

    impl crate::TrainTrackerBinary for TrainTrackerImpl {
        fn wait_for_cancelled(&self) -> BoxFuture<'_, ()> {
            Box::pin(std::future::pending())
        }

        fn progress(&self, progress: f32) -> BoxFuture<'_, ()> {
            Box::pin(async move {
                unsafe { dt_training_progress(progress) };
            })
        }

        fn metrics<'b>(
            &'b self,
            metrics: &'b [crate::MetricBinary<impl AsRef<str> + Sync + 'b>],
        ) -> BoxFuture<'b, ()> {
            Box::pin(async move {
                let mut buf = Vec::with_capacity(metrics.len() * 32);
                for metric in metrics {
                    buf.extend_from_slice(&(metric.name.as_ref().len() as u64).to_be_bytes());
                    buf.extend_from_slice(&(metric.name.as_ref().as_ptr() as u64).to_be_bytes());
                    buf.extend_from_slice(&(metric.data.as_ref().len() as u64).to_be_bytes());
                    buf.extend_from_slice(&(metric.data.as_ptr() as u64).to_be_bytes());
                }
                unsafe { dt_training_metrics(metrics.len() as u32, buf.as_ptr() as u64) };
            })
        }
    }

    pub fn get_train_tracker() -> impl crate::TrainTrackerBinary {
        TrainTrackerImpl
    }

    struct StateProviderImpl {
        provided: HashSet<String>,
    }

    impl crate::StateProvider for StateProviderImpl {
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

                    let mut ptrs = Vec::with_capacity(names.len() * 16);
                    for (name, data) in names.iter().zip(to_send) {
                        ptrs.extend_from_slice(&(name.len() as u64).to_be_bytes());
                        ptrs.extend_from_slice(&(name.as_ptr() as u64).to_be_bytes());
                        ptrs.extend_from_slice(&(data.len() as u64).to_be_bytes());
                        ptrs.extend_from_slice(&(data.as_ptr() as u64).to_be_bytes());
                    }
                    unsafe { dt_provide_state_data(names.len() as u32, ptrs.as_ptr() as u64) };
                }
            })
        }
    }

    pub fn get_state_provider() -> impl crate::StateProvider {
        StateProviderImpl {
            provided: HashSet::new(),
        }
    }

    pub unsafe fn read_other_models_with_state(
        ptr: u64,
    ) -> HashMap<String, crate::OtherModelWithState<impl crate::StateLoader>> {
        let num_other_models = u32::from_be_bytes(*(ptr as *const [u8; 4])) as usize;
        let mut pos = ptr + 4;

        let mut res = HashMap::with_capacity(num_other_models);
        for _ in 0..num_other_models {
            let id_len = u64::from_be_bytes(*(pos as *const [u8; 8])) as usize;
            pos += 8;
            let id_ptr = u64::from_be_bytes(*(pos as *const [u8; 8]));
            pos += 8;
            let mount_path_len = u64::from_be_bytes(*(pos as *const [u8; 8])) as usize;
            pos += 8;
            let mount_path_ptr = u64::from_be_bytes(*(pos as *const [u8; 8]));
            pos += 8;
            let state_ptr = u64::from_be_bytes(*(pos as *const [u8; 8]));
            pos += 8;
            let state = read_params(state_ptr);
            res.insert(
                String::from_utf8_unchecked(Vec::from_raw_parts(id_ptr as *mut u8, id_len, id_len)),
                crate::OtherModelWithState {
                    mount_path: String::from_utf8_unchecked(Vec::from_raw_parts(
                        mount_path_ptr as *mut u8,
                        mount_path_len,
                        mount_path_len,
                    )),
                    state,
                },
            );
        }
        res
    }

    pub unsafe fn read_other_models(ptr: u64) -> HashMap<String, crate::OtherModel> {
        let num_other_models = u32::from_be_bytes(*(ptr as *const [u8; 4])) as usize;
        let mut pos = ptr + 4;

        let mut res = HashMap::with_capacity(num_other_models);
        for _ in 0..num_other_models {
            let id_len = u64::from_be_bytes(*(pos as *const [u8; 8])) as usize;
            pos += 8;
            let id_ptr = u64::from_be_bytes(*(pos as *const [u8; 8]));
            pos += 8;
            let mount_path_len = u64::from_be_bytes(*(pos as *const [u8; 8])) as usize;
            pos += 8;
            let mount_path_ptr = u64::from_be_bytes(*(pos as *const [u8; 8]));
            pos += 8;
            res.insert(
                String::from_utf8_unchecked(Vec::from_raw_parts(id_ptr as *mut u8, id_len, id_len)),
                crate::OtherModel {
                    mount_path: String::from_utf8_unchecked(Vec::from_raw_parts(
                        mount_path_ptr as *mut u8,
                        mount_path_len,
                        mount_path_len,
                    )),
                },
            );
        }
        res
    }
}

pub use decthings_model_macros::*;
