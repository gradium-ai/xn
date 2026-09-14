use crate::{Backend, Result, Shape, Tensor, WithDTypeF};
use std::sync::{Arc, Mutex};

pub struct MmapedFiles {
    mmaps: Vec<(std::path::PathBuf, memmap2::Mmap)>,
}

impl MmapedFiles {
    pub fn load_from_files<P: AsRef<std::path::Path>>(file_paths: &[P]) -> Result<Self> {
        let mut mmaps = Vec::new();
        for path in file_paths {
            let path = path.as_ref();
            let file =
                std::fs::File::open(path).map_err(|e| crate::Error::from(e).with_path(path))?;
            let mmap = unsafe {
                memmap2::MmapOptions::new()
                    .map(&file)
                    .map_err(|e| crate::Error::from(e).with_path(path))?
            };
            mmaps.push((path.to_path_buf(), mmap));
        }
        Ok(Self { mmaps })
    }
}

#[derive(yoke::Yokeable)]
struct TensorData<'a> {
    data: &'a [u8],
    shape: Shape,
    dtype: crate::DType,
}

pub struct VarBuilder<'a, B: Backend> {
    tensor_data: std::collections::HashMap<String, TensorData<'a>>,
    device: B,
}

fn load_tensor_data(
    mmaps: &MmapedFiles,
) -> Result<std::collections::HashMap<String, TensorData<'_>>> {
    load_tensor_data_with_key_map(mmaps, |name| Some(name.to_string()))
}

fn load_tensor_data_with_key_map(
    mmaps: &MmapedFiles,
    key_map: impl Fn(&str) -> Option<String>,
) -> Result<std::collections::HashMap<String, TensorData<'_>>> {
    let mut tensor_data = std::collections::HashMap::new();
    for (path, mmap) in mmaps.mmaps.iter() {
        let tensors = safetensors::SafeTensors::deserialize(mmap)
            .map_err(|e| crate::Error::from(e).with_path(path))?;
        for (name, tensor) in tensors.iter() {
            let mapped_name = match key_map(name) {
                Some(n) => n,
                None => continue,
            };
            let shape: Shape = tensor.shape().into();
            let data = tensor.data();
            let dtype = match tensor.dtype() {
                safetensors::Dtype::F32 => crate::DType::F32,
                safetensors::Dtype::F16 => crate::DType::F16,
                safetensors::Dtype::BF16 => crate::DType::BF16,
                _ => continue,
            };
            let td = TensorData { data, shape, dtype };
            tensor_data.insert(mapped_name, td);
        }
    }
    Ok(tensor_data)
}

fn load_tensor_data_from_bytes(
    buffers: &[Vec<u8>],
) -> Result<std::collections::HashMap<String, TensorData<'_>>> {
    load_tensor_data_from_bytes_with_key_map(buffers, |name| Some(name.to_string()))
}

fn load_tensor_data_from_bytes_with_key_map(
    buffers: &[Vec<u8>],
    key_map: impl Fn(&str) -> Option<String>,
) -> Result<std::collections::HashMap<String, TensorData<'_>>> {
    let mut tensor_data = std::collections::HashMap::new();
    for buffer in buffers {
        let tensors = safetensors::SafeTensors::deserialize(buffer)?;
        for (name, tensor) in tensors.iter() {
            let mapped_name = match key_map(name) {
                Some(n) => n,
                None => continue,
            };
            let shape: Shape = tensor.shape().into();
            let data = tensor.data();
            let dtype = match tensor.dtype() {
                safetensors::Dtype::F32 => crate::DType::F32,
                safetensors::Dtype::F16 => crate::DType::F16,
                safetensors::Dtype::BF16 => crate::DType::BF16,
                _ => continue,
            };
            let td = TensorData { data, shape, dtype };
            tensor_data.insert(mapped_name, td);
        }
    }
    Ok(tensor_data)
}

impl<'a, B: Backend> VarBuilder<'a, B> {
    pub fn load(mmaped_files: &'a MmapedFiles, device: B) -> Result<Self> {
        let tensor_data = load_tensor_data(mmaped_files)?;
        Ok(Self { tensor_data, device })
    }

    pub fn device(&self) -> &B {
        &self.device
    }

    pub fn tensor<T: WithDTypeF>(
        &self,
        name: &str,
        shape: impl Into<Shape>,
    ) -> Result<Tensor<T, B>> {
        let td = self.tensor_data.get(name);
        make_tensor(td, name, shape, &self.device)
    }
}

fn make_tensor<T: WithDTypeF, B: Backend>(
    td: Option<&TensorData<'_>>,
    name: &str,
    shape: impl Into<Shape>,
    device: &B,
) -> Result<Tensor<T, B>> {
    let td = match td {
        Some(t) => t,
        None => crate::bail!("tensor '{name}' not found"),
    };
    let shape = shape.into();
    if td.shape != shape {
        crate::bail!(
            "shape mismatch for tensor '{name}': expected {shape:?}, found {:?}",
            td.shape
        );
    }
    let data = crate::dtype::convert_bytes_to_vec::<T>(td.data, td.dtype);
    let tensor = Tensor::from_vec(data, shape, device)?;
    Ok(tensor)
}

// Inner yokeable struct that holds the borrowed tensor data
#[derive(yoke::Yokeable)]
struct VarBuilderYoke<'a> {
    tensor_data: std::collections::HashMap<String, TensorData<'a>>,
}

#[derive(yoke::Yokeable)]
struct VarBuilderYokeBytes<'a> {
    tensor_data: std::collections::HashMap<String, TensorData<'a>>,
}

pub trait Reader: std::io::Seek + std::io::Read {}

enum VBData {
    Mmap(yoke::Yoke<VarBuilderYoke<'static>, Box<MmapedFiles>>),
    Bytes(yoke::Yoke<VarBuilderYokeBytes<'static>, Vec<Vec<u8>>>),
    Gguf(crate::quantized::gguf_file::Content, Mutex<Box<dyn Reader>>),
}

impl VBData {
    fn tensor_names(&self) -> Vec<&str> {
        match self {
            Self::Mmap(yoke) => yoke.get().tensor_data.keys().map(|k| k.as_str()).collect(),
            Self::Bytes(yoke) => yoke.get().tensor_data.keys().map(|k| k.as_str()).collect(),
            Self::Gguf(content, _) => content.tensor_infos.keys().map(|k| k.as_str()).collect(),
        }
    }

    fn contains(&self, name: &str) -> bool {
        match self {
            Self::Mmap(yoke) => yoke.get().tensor_data.contains_key(name),
            Self::Bytes(yoke) => yoke.get().tensor_data.contains_key(name),
            Self::Gguf(content, _) => content.tensor_infos.contains_key(name),
        }
    }

    fn shape(&self, name: &str) -> Option<&Shape> {
        match self {
            Self::Mmap(yoke) => yoke.get().tensor_data.get(name).map(|td| &td.shape),
            Self::Bytes(yoke) => yoke.get().tensor_data.get(name).map(|td| &td.shape),
            Self::Gguf(content, _) => content.tensor_infos.get(name).map(|info| &info.shape),
        }
    }
}

impl Reader for std::io::BufReader<std::fs::File> {}
impl Reader for std::io::Cursor<Vec<u8>> {}

/// A self-contained VarBuilder that owns its data (memory-mapped files or byte buffers).
pub struct VB<B: Backend> {
    data: VBData,
    used: Mutex<std::collections::HashSet<String>>,
    device: B,
}

impl<B: Backend> VB<B> {
    pub fn load<P: AsRef<std::path::Path>>(file_paths: &[P], device: B) -> Result<Self> {
        let mmaps = MmapedFiles::load_from_files(file_paths)?;
        let yoke = yoke::Yoke::try_attach_to_cart(Box::new(mmaps), |mmaps| -> Result<_> {
            let tensor_data = load_tensor_data(mmaps)?;
            Ok(VarBuilderYoke { tensor_data })
        })?;
        let used = Mutex::new(Default::default());
        Ok(Self { data: VBData::Mmap(yoke), used, device })
    }

    pub fn load_gguf<R: Reader + 'static>(mut reader: R, device: B) -> Result<Self> {
        let content = crate::quantized::gguf_file::Content::read(&mut reader)?;
        let reader = Mutex::new(Box::new(reader) as Box<dyn Reader>);
        let data = VBData::Gguf(content, reader);
        let used = Mutex::new(Default::default());
        Ok(Self { data, used, device })
    }

    pub fn load_with_key_map<P: AsRef<std::path::Path>>(
        file_paths: &[P],
        device: B,
        key_map: impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        let mmaps = MmapedFiles::load_from_files(file_paths)?;
        let yoke = yoke::Yoke::try_attach_to_cart(Box::new(mmaps), |mmaps| -> Result<_> {
            let tensor_data = load_tensor_data_with_key_map(mmaps, &key_map)?;
            Ok(VarBuilderYoke { tensor_data })
        })?;
        let used = Mutex::new(Default::default());
        Ok(Self { data: VBData::Mmap(yoke), used, device })
    }

    pub fn load_gguf_with_key_map<R: Reader + 'static>(
        mut reader: R,
        device: B,
        key_map: impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        let content = crate::quantized::gguf_file::Content::read(&mut reader)?;
        let content = content.apply_key_map(key_map);
        let reader = Mutex::new(Box::new(reader) as Box<dyn Reader>);
        let data = VBData::Gguf(content, reader);
        let used = Mutex::new(Default::default());
        Ok(Self { data, used, device })
    }

    pub fn from_bytes(data: Vec<Vec<u8>>, device: B) -> Result<Self> {
        let yoke = yoke::Yoke::try_attach_to_cart(data, |buffers| -> Result<_> {
            let tensor_data = load_tensor_data_from_bytes(buffers)?;
            Ok(VarBuilderYokeBytes { tensor_data })
        })?;
        let used = Mutex::new(Default::default());
        Ok(Self { data: VBData::Bytes(yoke), used, device })
    }

    pub fn from_bytes_with_key_map(
        data: Vec<Vec<u8>>,
        device: B,
        key_map: impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        let yoke = yoke::Yoke::try_attach_to_cart(data, |buffers| -> Result<_> {
            let tensor_data = load_tensor_data_from_bytes_with_key_map(buffers, &key_map)?;
            Ok(VarBuilderYokeBytes { tensor_data })
        })?;
        let used = Mutex::new(Default::default());
        Ok(Self { data: VBData::Bytes(yoke), used, device })
    }

    pub fn device(&self) -> &B {
        &self.device
    }

    /// Returns a quantized tensor if the underlying data is from a GGUF file, otherwise returns
    /// None.
    pub fn qtensor(&self, name: &str) -> Result<Option<crate::quantized::QTensor>> {
        match &self.data {
            VBData::Mmap(_) | VBData::Bytes(_) => Ok(None),
            VBData::Gguf(content, reader) => {
                let tensor = {
                    let mut reader = reader.lock().unwrap();
                    content.tensor(&mut *reader, name)?
                };
                {
                    let mut t = self.used.lock().unwrap();
                    t.insert(name.to_string());
                }
                Ok(Some(tensor))
            }
        }
    }

    pub fn tensor<T: WithDTypeF>(
        &self,
        name: &str,
        shape: impl Into<Shape>,
    ) -> Result<Tensor<T, B>> {
        match &self.data {
            VBData::Mmap(yoke) => {
                let td = yoke.get().tensor_data.get(name);
                if td.is_some() {
                    let mut t = self.used.lock().unwrap();
                    t.insert(name.to_string());
                }
                make_tensor(td, name, shape, &self.device)
            }
            VBData::Bytes(yoke) => {
                let td = yoke.get().tensor_data.get(name);
                if td.is_some() {
                    let mut t = self.used.lock().unwrap();
                    t.insert(name.to_string());
                }
                make_tensor(td, name, shape, &self.device)
            }
            VBData::Gguf(content, reader) => {
                let tensor = {
                    let mut reader = reader.lock().unwrap();
                    content.tensor(&mut *reader, name)?
                };
                {
                    let mut t = self.used.lock().unwrap();
                    t.insert(name.to_string());
                }
                let shape = tensor.shape();
                let dequantized = tensor.dequantize()?;
                Tensor::from_vec(dequantized, shape, &self.device)?.to()
            }
        }
    }

    /// A tensor's values on the host, without ever putting it on the device.
    ///
    /// Some weights are only ever needed host-side -- the flow LM's `bos_emb` is
    /// compared against every frame on the CPU -- and loading them as a tensor
    /// only to read them straight back is a pointless round trip. On a GPU
    /// backend it is worse than pointless: in a browser a blocking readback is
    /// impossible, so that round trip cannot happen at all.
    pub fn tensor_host<T: WithDTypeF>(
        &self,
        name: &str,
        shape: impl Into<Shape>,
    ) -> Result<Vec<T>> {
        let shape = shape.into();
        let from_td = |td: Option<&TensorData<'_>>| -> Result<Vec<T>> {
            let td = match td {
                Some(t) => t,
                None => crate::bail!("tensor '{name}' not found"),
            };
            if td.shape != shape {
                crate::bail!(
                    "shape mismatch for tensor '{name}': expected {shape:?}, found {:?}",
                    td.shape
                );
            }
            Ok(crate::dtype::convert_bytes_to_vec::<T>(td.data, td.dtype))
        };
        match &self.data {
            VBData::Mmap(yoke) => {
                let out = from_td(yoke.get().tensor_data.get(name))?;
                self.used.lock().unwrap().insert(name.to_string());
                Ok(out)
            }
            VBData::Bytes(yoke) => {
                let out = from_td(yoke.get().tensor_data.get(name))?;
                self.used.lock().unwrap().insert(name.to_string());
                Ok(out)
            }
            VBData::Gguf(content, reader) => {
                let tensor = {
                    let mut reader = reader.lock().unwrap();
                    content.tensor(&mut *reader, name)?
                };
                self.used.lock().unwrap().insert(name.to_string());
                if tensor.shape() != &shape {
                    crate::bail!(
                        "shape mismatch for tensor '{name}': expected {shape:?}, found {:?}",
                        tensor.shape()
                    );
                }
                Ok(tensor.dequantize()?.into_iter().map(T::from_f32).collect())
            }
        }
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.data.tensor_names()
    }

    pub fn root(self) -> Path<B> {
        Path { vb: self.into(), path: vec![] }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.data.contains(name)
    }

    pub fn shape(&self, name: &str) -> Option<&Shape> {
        self.data.shape(name)
    }

    pub fn check_all_used(&self) -> Result<()> {
        self.check_all_used_with_ignore(|_| false)
    }

    pub fn check_all_used_with_ignore(&self, ignore_f: impl Fn(&str) -> bool) -> Result<()> {
        let used = self.used.lock().unwrap();
        let mut unused = vec![];
        for tensor_name in self.tensor_names() {
            if !used.contains(tensor_name) && !ignore_f(tensor_name) {
                unused.push(tensor_name);
            }
        }
        if !unused.is_empty() {
            unused.sort();
            crate::bail!("{} unused tensors {unused:?}", unused.len())
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Path<B: Backend> {
    path: Vec<String>,
    vb: Arc<VB<B>>,
}

impl<B: Backend> Path<B> {
    pub fn device(&self) -> &B {
        self.vb.device()
    }

    pub fn qtensor(&self, name: &str) -> Result<Option<crate::quantized::QTensor>> {
        let name = self.path(name);
        self.vb.qtensor(&name)
    }

    pub fn tensor<T: WithDTypeF>(
        &self,
        name: &str,
        shape: impl Into<Shape>,
    ) -> Result<Tensor<T, B>> {
        let name = self.path(name);
        self.vb.tensor(&name, shape)
    }

    /// See [`VB::tensor_host`]: the values on the host, never on the device.
    pub fn tensor_host<T: WithDTypeF>(
        &self,
        name: &str,
        shape: impl Into<Shape>,
    ) -> Result<Vec<T>> {
        let name = self.path(name);
        self.vb.tensor_host(&name, shape)
    }

    /// Return a new `VarBuilder` adding `s` to the current prefix. This can be think of as `cd`
    /// into a directory.
    pub fn push_prefix<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self { vb: self.vb.clone(), path }
    }

    /// Short alias for `push_prefix`.
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        self.push_prefix(s)
    }

    /// Returns the prefix of the `VarBuilder`.
    pub fn prefix(&self) -> String {
        self.path.join(".")
    }

    /// Check if a tensor with the given name exists.
    pub fn contains(&self, name: &str) -> bool {
        let name = self.path(name);
        self.vb.data.contains(&name)
    }

    pub fn shape(&self, name: &str) -> Option<&Shape> {
        let name = self.path(name);
        self.vb.data.shape(&name)
    }

    fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    pub fn check_all_used(&self) -> Result<()> {
        self.vb.check_all_used()
    }

    pub fn check_all_used_with_ignore(&self, ignore_f: impl Fn(&str) -> bool) -> Result<()> {
        self.vb.check_all_used_with_ignore(ignore_f)
    }
}
