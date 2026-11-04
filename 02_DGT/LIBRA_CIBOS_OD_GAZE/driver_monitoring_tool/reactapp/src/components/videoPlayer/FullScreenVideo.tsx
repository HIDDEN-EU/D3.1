import { type Ref } from "react";
import { Spin, Typography } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import VideoPlayer from "./VideoPlayer";
import { useAutoHideOnFullscreen } from "../../hooks/useAutoHideOnFullscreen";
import FullscreenHoverBanner from "../gazeTracking/FullscreenHoverBanner";

const { Text } = Typography;

interface FullScreenVideoProps {
    streamID: number;
    mediaID: string;
    isFullscreen: boolean;
    videoRef: Ref<HTMLInputElement>;
    isLoadingMountpoint: boolean;
    onCancel: () => void;
}

const FullScreenVideo: React.FC<FullScreenVideoProps> = ({
    streamID,
    mediaID,
    isFullscreen, 
    videoRef,
    isLoadingMountpoint,
    onCancel
}) => {
    const controlsVisible = useAutoHideOnFullscreen(videoRef as React.RefObject<HTMLElement>, isFullscreen);
    
    return <>
        <div 
            ref={videoRef}
            className={isFullscreen ? "video-wrapper" : "hidden-video"} 
            style={{backgroundColor: "#bbb8b8", position: "relative", cursor: isFullscreen && !controlsVisible ? "none" : "default" }}
        >
            {isFullscreen && <FullscreenHoverBanner visible={controlsVisible} onCancel={onCancel} />}
            {isLoadingMountpoint
                ? <div style={{display: "flex", justifyContent: "center", alignItems: "center", flex: 2}}>
                        <span>
                            <Text style={{marginRight: "15px", fontSize: "17px"}}>
                                Loading stream mountpoint...
                            </Text>
                            <Spin size="large" indicator={<LoadingOutlined />} /> 
                        </span>
                    </div>
                : <div style={{ width: "100%", height: "100%"}}>
                    {isFullscreen && <VideoPlayer
                        streamID={streamID}
                        mediaID={mediaID}
                    />}
                </div>
            }
        </div>
    </>
};

export default FullScreenVideo;