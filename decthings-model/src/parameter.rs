#[derive(Debug, Clone)]
pub struct Parameter<'a> {
    pub name: String,
    pub data: Vec<decthings_api::tensor::DecthingsTensor<'a>>,
}

#[cfg_attr(target_family = "unix", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct ParameterBinary {
    pub name: String,
    #[cfg_attr(target_family = "unix", serde(skip_serializing, skip_deserializing))]
    pub data: Vec<bytes::Bytes>,
}

impl<'a> From<Parameter<'a>> for ParameterBinary {
    fn from(value: Parameter) -> Self {
        Self {
            name: value.name,
            data: value
                .data
                .into_iter()
                .map(|x| x.serialize().into())
                .collect(),
        }
    }
}
