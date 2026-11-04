import React from "react";
import JanusVideoPlayer from "./JanusVideoPlayer";

interface VideoPlayerProps {
    streamID: number;
    mediaID: string;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({
    streamID,
    mediaID,
}) => {
    
    return <div style={{ width: "100%", height: "100%"}}>
        <JanusVideoPlayer
            streamID={streamID}
            mediaID={mediaID}
        />
    </div>;
};

export default React.memo(VideoPlayer);