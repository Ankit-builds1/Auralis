from aiortc import RTCPeerConnection


class WebRTCConnection:
    """
    Manages one WebRTC peer connection.
    """

    def __init__(self):
        self.pc = RTCPeerConnection()

    async def close(self):
        await self.pc.close()