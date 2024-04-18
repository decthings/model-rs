pub use tokio::{
    io::Error,
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader, BufWriter},
    net::UnixStream,
    sync::mpsc::{channel, Receiver, Sender},
    sync::oneshot,
    task::spawn,
};

pub fn unix_split(
    stream: &mut UnixStream,
) -> (impl AsyncRead + Unpin + '_, impl AsyncWrite + Unpin + '_) {
    stream.split()
}

pub async fn channel_recv<T>(channel: &mut Receiver<T>) -> Option<T> {
    channel.recv().await
}

pub async fn read_u8(mut reader: impl AsyncRead + Unpin) -> Result<u8, Error> {
    let mut buf = [0; 1];
    reader.read_exact(&mut buf).await?;
    Ok(u8::from_be_bytes(buf))
}

pub async fn write_u8(mut writer: impl AsyncWrite + Unpin, value: u8) -> Result<(), Error> {
    writer.write_all(&value.to_be_bytes()).await
}

pub async fn read_u32(mut reader: impl AsyncRead + Unpin) -> Result<u32, Error> {
    let mut buf = [0; 4];
    reader.read_exact(&mut buf).await?;
    Ok(u32::from_be_bytes(buf))
}

pub async fn write_u32(mut writer: impl AsyncWrite + Unpin, value: u32) -> Result<(), Error> {
    writer.write_all(&value.to_be_bytes()).await
}

pub async fn read_u64(mut reader: impl AsyncRead + Unpin) -> Result<u64, Error> {
    let mut buf = [0; 8];
    reader.read_exact(&mut buf).await?;
    Ok(u64::from_be_bytes(buf))
}

pub async fn write_u64(mut writer: impl AsyncWrite + Unpin, value: u64) -> Result<(), Error> {
    writer.write_all(&value.to_be_bytes()).await
}
