from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer


class WebRTCConnection:
    """
    Manages one WebRTC peer connection.
    """

    def __init__(self):
        self.pc = RTCPeerConnection(
            configuration=RTCConfiguration(
                iceServers=[
                    RTCIceServer(urls="stun:stun.l.google.com:19302"),
                    RTCIceServer(
                        urls="turn:openrelay.metered.ca:80",
                        username="openrelayproject",
                        credential="openrelayproject"
                    )
                ]
            )
        )

    async def close(self):
        await self.pc.close()