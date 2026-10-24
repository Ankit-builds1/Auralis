import os

from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer


def ice_servers():
    """STUN always; a TURN relay when TURN_URL/TURN_USERNAME/TURN_CREDENTIAL
    are set in .env (deployment: lets browsers on other networks reach this
    laptop). TURN_URL may hold several URLs separated by commas."""
    servers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
    urls = [u.strip() for u in os.getenv("TURN_URL", "").split(",") if u.strip()]
    user = os.getenv("TURN_USERNAME", "")
    cred = os.getenv("TURN_CREDENTIAL", "")
    if urls and user and cred:
        servers.append(RTCIceServer(urls=urls, username=user, credential=cred))
    return servers


class WebRTCConnection:
    """
    Manages one WebRTC peer connection.
    """

    def __init__(self):
        self.pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=ice_servers())
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