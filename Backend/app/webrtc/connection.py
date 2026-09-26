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

        # WebRTC ICE connection state
        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            print(
                f"[WebRTC] ICE connection state: "
                f"{self.pc.iceConnectionState}"
            )

        # Overall WebRTC connection state
        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(
                f"[WebRTC] Connection state: "
                f"{self.pc.connectionState}"
            )

        # ICE gathering state
        @self.pc.on("icegatheringstatechange")
        async def on_icegatheringstatechange():
            print(
                f"[WebRTC] ICE gathering state: "
                f"{self.pc.iceGatheringState}"
            )

    async def close(self):
        await self.pc.close()