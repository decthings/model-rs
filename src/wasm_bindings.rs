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

            pub struct WeightsLoaderImpl {
                pub byte_size: u64,
                pub inner: super::$($path_to_types_root)*::exports::decthings::model::model::WeightsLoader,
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

            fn next(&mut self, mut amount: u32) -> ::core::pin::Pin<::std::boxed::Box<dyn ::core::future::Future<Output = ::std::vec::Vec<::decthings_model::bytes::Bytes>> + Send + '_>> {
                amount = amount.min(::decthings_model::DataLoaderBinary::remaining(self));

                let start_index = self.position;
                self.position += amount;
                ::std::boxed::Box::pin(async move {
                    if amount == 0 {
                        return vec![];
                    }
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

        impl ::decthings_model::WeightsProvider for $($path_to_types_root)*::exports::decthings::model::model::WeightsProvider {
            fn provide_all<'a>(
                &'a mut self,
                data: &'a [(impl AsRef<str> + Send + Sync + 'a, ::decthings_model::bytes::Bytes)],
            ) -> ::core::pin::Pin<::std::boxed::Box<dyn ::core::future::Future<Output = ()> + Send + 'a>> {
                ::std::boxed::Box::pin(async move {
                    $($path_to_types_root)*::exports::decthings::model::model::WeightsProvider::provide(
                        self,
                        &data.into_iter().map(|data| (data.0.as_ref().to_owned(), data.1.to_vec())).collect::<::std::vec::Vec<_>>()
                    );
                })
            }
        }

        impl ::decthings_model::WeightsLoader for _decthings_model::WeightsLoaderImpl {
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
                                expected_output_types: options.expected_output_types.into_iter().map(|output| (
                                    output.name,
                                    ::decthings_model::ExpectedOutputType {
                                        required: output.required,
                                        rules: ::decthings_model::decthings_api::tensor::DecthingsTensorRules {
                                            shape: output.rules.shape,
                                            allowed_types: output
                                                .rules
                                                .allowed_types
                                                .into_iter()
                                                .map(|y| match y {
                                                    $($path_to_types_root)*::exports::decthings::model::model::DecthingsElementType::F32 => {
                                                        ::decthings_model::decthings_api::tensor::DecthingsElementType::F32
                                                    }
                                                    $($path_to_types_root)*::exports::decthings::model::model::DecthingsElementType::F64 => {
                                                        ::decthings_model::decthings_api::tensor::DecthingsElementType::F64
                                                    }
                                                    $($path_to_types_root)*::exports::decthings::model::model::DecthingsElementType::I8 => {
                                                        ::decthings_model::decthings_api::tensor::DecthingsElementType::I8
                                                    }
                                                    $($path_to_types_root)*::exports::decthings::model::model::DecthingsElementType::I16 => {
                                                        ::decthings_model::decthings_api::tensor::DecthingsElementType::I16
                                                    }
                                                    $($path_to_types_root)*::exports::decthings::model::model::DecthingsElementType::I32 => {
                                                        ::decthings_model::decthings_api::tensor::DecthingsElementType::I32
                                                    }
                                                    $($path_to_types_root)*::exports::decthings::model::model::DecthingsElementType::I64 => {
                                                        ::decthings_model::decthings_api::tensor::DecthingsElementType::I64
                                                    }
                                                    $($path_to_types_root)*::exports::decthings::model::model::DecthingsElementType::U8 => {
                                                        ::decthings_model::decthings_api::tensor::DecthingsElementType::U8
                                                    }
                                                    $($path_to_types_root)*::exports::decthings::model::model::DecthingsElementType::U16 => {
                                                        ::decthings_model::decthings_api::tensor::DecthingsElementType::U16
                                                    }
                                                    $($path_to_types_root)*::exports::decthings::model::model::DecthingsElementType::U32 => {
                                                        ::decthings_model::decthings_api::tensor::DecthingsElementType::U32
                                                    }
                                                    $($path_to_types_root)*::exports::decthings::model::model::DecthingsElementType::U64 => {
                                                        ::decthings_model::decthings_api::tensor::DecthingsElementType::U64
                                                    }
                                                    $($path_to_types_root)*::exports::decthings::model::model::DecthingsElementType::String => {
                                                        ::decthings_model::decthings_api::tensor::DecthingsElementType::String
                                                    }
                                                    $($path_to_types_root)*::exports::decthings::model::model::DecthingsElementType::Boolean => {
                                                        ::decthings_model::decthings_api::tensor::DecthingsElementType::Boolean
                                                    }
                                                    $($path_to_types_root)*::exports::decthings::model::model::DecthingsElementType::Binary => {
                                                        ::decthings_model::decthings_api::tensor::DecthingsElementType::Binary
                                                    }
                                                    $($path_to_types_root)*::exports::decthings::model::model::DecthingsElementType::Image => {
                                                        ::decthings_model::decthings_api::tensor::DecthingsElementType::Image
                                                    }
                                                    $($path_to_types_root)*::exports::decthings::model::model::DecthingsElementType::Audio => {
                                                        ::decthings_model::decthings_api::tensor::DecthingsElementType::Audio
                                                    }
                                                    $($path_to_types_root)*::exports::decthings::model::model::DecthingsElementType::Video => {
                                                        ::decthings_model::decthings_api::tensor::DecthingsElementType::Video
                                                    }
                                                })
                                                .collect(),
                                            annotations: vec![],
                                        }
                                    }
                                )).collect()
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

            fn get_weights(
                &self,
                options: $($path_to_types_root)*::exports::decthings::model::model::GetWeightsOptions,
            ) -> Result<(), String> {
                ::decthings_model::wasm_bindings::pollster::block_on(
                    T::get_weights(
                        self,
                        ::decthings_model::GetWeightsOptions {
                            weights_provider: options.weights_provider,
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

            fn initialize_weights(
                options: $($path_to_types_root)*::exports::decthings::model::model::InitializeWeightsOptions,
            ) -> Result<(), String> {
                ::decthings_model::wasm_bindings::pollster::block_on(
                    T::initialize_weights(
                        ::decthings_model::InitializeWeightsOptions {
                            params: options.params.into_iter().map(|param| (
                                param.name,
                                _decthings_model::DataLoaderBinaryImpl {
                                    position: 0,
                                    amount: param.amount,
                                    total_byte_size: param.total_byte_size,
                                    inner: param.data_loader,
                                },
                            )).collect(),
                            weights_provider: options.weights_provider,
                            other_models: options.other_models.into_iter()
                                .map(|other_model| (
                                    other_model.model_id,
                                    ::decthings_model::OtherModelWithWeights {
                                        mount_path: other_model.mount_path,
                                        weights: other_model.weights.into_iter().map(|weight_key| (
                                            weight_key.key,
                                            _decthings_model::WeightsLoaderImpl {
                                                byte_size: weight_key.byte_size,
                                                inner: weight_key.weights_loader,
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
                                weights: options.weights.into_iter().map(|weight_key| (
                                    weight_key.key,
                                    _decthings_model::WeightsLoaderImpl {
                                        byte_size: weight_key.byte_size,
                                        inner: weight_key.weights_loader,
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
