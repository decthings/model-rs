pub use pollster;

#[macro_export]
macro_rules! export_decthings_model {
    ($ty:ident with_types_in $($path_to_types_root:tt)*) => {
        mod _decthings_model {
            pub struct DataLoaderBinaryImpl {
                pub position: u32,
                pub amount: u32,
                pub total_byte_size: u64,
                pub inner: super::$($path_to_types_root)*::exports::decthings::model::model::DataLoader,
            }

            pub struct StateLoaderImpl {
                pub byte_size: u64,
                pub inner: super::$($path_to_types_root)*::exports::decthings::model::model::StateLoader,
            }
        }

        impl ::decthings_model::DataLoaderBinary for _decthings_model::DataLoaderBinaryImpl {
            fn total_byte_size(&self) -> u64 {
                self.total_byte_size
            }

            fn shuffle_in_group<'a>(&'a self, others: &'a [&'a Self]) -> ::core::pin::Pin<::std::boxed::Box<dyn ::core::future::Future<Output = ()> + Send + 'a>> {
                ::std::boxed::Box::pin(async move {
                    self.inner.shuffle(&others.iter().map(|x| &x.inner).collect::<::std::vec::Vec<_>>());
                })
            }

            fn size(&self) -> u32 {
                self.amount
            }

            fn position(&self) -> u32 {
                self.position
            }

            fn set_position(&mut self, position: u32) {
                self.position = position;
            }

            fn next(&mut self, amount: u32) -> ::core::pin::Pin<::std::boxed::Box<dyn ::core::future::Future<Output = ::std::vec::Vec<::decthings_model::bytes::Bytes>> + Send + '_>> {
                let start_index = self.position;
                self.position += amount;
                ::std::boxed::Box::pin(async move {
                    self.inner.read(start_index, amount).into_iter().map(Into::into).collect()
                })
            }
        }

        impl ::decthings_model::TrainTrackerBinary for $($path_to_types_root)*::exports::decthings::model::model::TrainTracker {
            fn wait_for_cancelled(&self) -> ::core::pin::Pin<::std::boxed::Box<dyn ::core::future::Future<Output = ()> + Send + '_>> {
                ::std::boxed::Box::pin(::core::future::pending())
            }

            fn progress(&self, progress: f32) -> ::core::pin::Pin<::std::boxed::Box<dyn ::core::future::Future<Output = ()> + Send + '_>> {
                ::std::boxed::Box::pin(async move {
                    $($path_to_types_root)*::exports::decthings::model::model::TrainTracker::progress(
                        self,
                        progress,
                    );
                })
            }

            fn metrics<'a>(
                &'a self,
                metrics: &'a [::decthings_model::MetricBinary<impl AsRef<str> + Sync + 'a>],
            ) -> ::core::pin::Pin<::std::boxed::Box<dyn ::core::future::Future<Output = ()> + Send + '_>> {
                ::std::boxed::Box::pin(async move {
                    $($path_to_types_root)*::exports::decthings::model::model::TrainTracker::metrics(
                        self,
                        &metrics.iter().map(|metric| (metric.name.as_ref().to_owned(), metric.data.to_vec())).collect::<::std::vec::Vec<_>>()
                    );
                })
            }
        }

        impl ::decthings_model::StateProvider for $($path_to_types_root)*::exports::decthings::model::model::StateProvider {
            fn provide_all<'a>(
                &'a mut self,
                data: &'a [(impl AsRef<str> + Send + Sync + 'a, ::decthings_model::bytes::Bytes)],
            ) -> ::core::pin::Pin<::std::boxed::Box<dyn ::core::future::Future<Output = ()> + Send + 'a>> {
                ::std::boxed::Box::pin(async move {
                    $($path_to_types_root)*::exports::decthings::model::model::StateProvider::provide(
                        self,
                        &data.into_iter().map(|data| (data.0.as_ref().to_owned(), data.1.to_vec())).collect::<::std::vec::Vec<_>>()
                    );
                })
            }
        }

        impl ::decthings_model::StateLoader for _decthings_model::StateLoaderImpl {
            fn byte_size(&self) -> u64 {
                self.byte_size
            }

            fn read(&mut self) -> ::core::pin::Pin<::std::boxed::Box<dyn ::core::future::Future<Output = ::decthings_model::bytes::Bytes> + Send + '_>> {
                ::std::boxed::Box::pin(async move { self.inner.read().into() })
            }
        }

        impl<T: ::decthings_model::InstantiatedBinary + 'static> $($path_to_types_root)*::exports::decthings::model::model::GuestInstantiated for T {
            fn evaluate(
                &self,
                options: $($path_to_types_root)*::exports::decthings::model::model::EvaluateOptions,
            ) -> Result<::std::vec::Vec<$($path_to_types_root)*::exports::decthings::model::model::EvaluateOutput>, String> {
                Ok(
                    ::decthings_model::wasm_bindings::pollster::block_on(
                        T::evaluate(
                            self,
                            ::decthings_model::EvaluateOptions {
                                params: options.params.into_iter().map(|param| (
                                    param.name,
                                    _decthings_model::DataLoaderBinaryImpl {
                                        position: 0,
                                        amount: param.amount,
                                        total_byte_size: param.total_byte_size,
                                        inner: param.data_loader,
                                    },
                                )).collect(),
                            }
                        )
                    )
                        .into_iter()
                        .map(|output| $($path_to_types_root)*::exports::decthings::model::model::EvaluateOutput {
                            name: output.name,
                            data: output.data.into_iter().map(|x| x.to_vec()).collect(),
                        })
                        .collect()
                )
            }

            fn train(&self, options: $($path_to_types_root)*::exports::decthings::model::model::TrainOptions) -> Result<(), String> {
                ::decthings_model::wasm_bindings::pollster::block_on(
                    T::train(
                        self,
                        ::decthings_model::TrainOptions {
                            params: options.params.into_iter().map(|param| (
                                param.name,
                                _decthings_model::DataLoaderBinaryImpl {
                                    position: 0,
                                    amount: param.amount,
                                    total_byte_size: param.total_byte_size,
                                    inner: param.data_loader,
                                },
                            )).collect(),
                            tracker: options.tracker,
                        }
                    )
                );
                Ok(())
            }

            fn get_model_state(
                &self,
                options: $($path_to_types_root)*::exports::decthings::model::model::GetModelStateOptions,
            ) -> Result<(), String> {
                ::decthings_model::wasm_bindings::pollster::block_on(
                    T::get_model_state(
                        self,
                        ::decthings_model::GetModelStateOptions {
                            state_provider: options.state_provider,
                        }
                    )
                );
                Ok(())
            }
        }

        impl<T: ::decthings_model::ModelBinary> $($path_to_types_root)*::exports::decthings::model::model::Guest for T
            where T::Instantiated: 'static
        {
            type Instantiated = T::Instantiated;

            fn create_model_state(
                options: $($path_to_types_root)*::exports::decthings::model::model::CreateModelStateOptions,
            ) -> Result<(), String> {
                ::decthings_model::wasm_bindings::pollster::block_on(
                    T::create_model_state(
                        ::decthings_model::CreateModelStateOptions {
                            params: options.params.into_iter().map(|param| (
                                param.name,
                                _decthings_model::DataLoaderBinaryImpl {
                                    position: 0,
                                    amount: param.amount,
                                    total_byte_size: param.total_byte_size,
                                    inner: param.data_loader,
                                },
                            )).collect(),
                            state_provider: options.state_provider,
                            other_models: options.other_models.into_iter()
                                .map(|other_model| (
                                    other_model.model_id,
                                    ::decthings_model::OtherModelWithState {
                                        mount_path: other_model.mount_path,
                                        state: other_model.state.into_iter().map(|state_key| (
                                            state_key.key,
                                            _decthings_model::StateLoaderImpl {
                                                byte_size: state_key.byte_size,
                                                inner: state_key.state_loader,
                                            },
                                        )).collect(),
                                    },
                                ))
                                .collect()
                        }
                    )
                );
                Ok(())
            }

            fn instantiate_model(
                options: $($path_to_types_root)*::exports::decthings::model::model::InstantiateModelOptions,
            ) -> Result<$($path_to_types_root)*::exports::decthings::model::model::Instantiated, String> {
                Ok($($path_to_types_root)*::exports::decthings::model::model::Instantiated::new(
                    ::decthings_model::wasm_bindings::pollster::block_on(
                        T::instantiate_model(
                            ::decthings_model::InstantiateModelOptions {
                                state: options.state.into_iter().map(|state_key| (
                                    state_key.key,
                                    _decthings_model::StateLoaderImpl {
                                        byte_size: state_key.byte_size,
                                        inner: state_key.state_loader,
                                    },
                                )).collect(),
                                other_models: options.other_models.into_iter().map(|other_model| (
                                    other_model.model_id,
                                    ::decthings_model::OtherModel {
                                        mount_path: other_model.mount_path,
                                    },
                                )).collect()
                            }
                        )
                    )
                ))
            }
        }

        $($path_to_types_root)*::export!($ty with_types_in $($path_to_types_root)*);
    }
}

pub use export_decthings_model;
