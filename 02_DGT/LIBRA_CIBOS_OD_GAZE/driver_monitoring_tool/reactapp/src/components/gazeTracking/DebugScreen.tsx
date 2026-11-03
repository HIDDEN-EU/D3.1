import { Col, Row, Spin, Typography } from "antd";
import { useEffect, useState } from "react";
import VideoPlayer from "../videoPlayer/VideoPlayer";
import { LoadingOutlined } from "@ant-design/icons";
import CameraPredPlane from "./debugPanels/CameraPredPlane";
import type { JanusMountpoint } from "../../types/janusRelatedTypes";

const { Text } = Typography;

const DebugScreen: React.FC = () => {
    // const channel = new BroadcastChannel("debug");
    const [showEmotion, setShowEmotion] = useState<boolean>(false);
    const [showPerclos, setShowPerclos] = useState<boolean>(false);
    const [hasStarted, setHasStarted] = useState<boolean>(false);
    const [mode, setMode] = useState<"calibration" | "inference" | null>(null);
    const [mountpoints, setMountpoints] = useState<JanusMountpoint[]>([]);
    const [initializedMounts, setInitializedMounts] = useState<boolean>(false);     //flag for initialized mountpoints
    const [loading, setLoading] = useState<boolean>(true);
    const [streamInfo, setStreamInfo] = useState<Record<string, {streamID?: number; mediaID?: string;}>>({
        "jetsonCamera": {streamID: undefined, mediaID: undefined},
        "croppedFace": {streamID: undefined, mediaID: undefined},
        "croppedEye1": {streamID: undefined, mediaID: undefined},
        "croppedEye2": {streamID: undefined, mediaID: undefined},
    });
    //---ROS topics---
    const cameraTopic = mode==="calibration" ? "/dgt/gaze/raw" : mode==="inference" ? "/dgt/gaze/mapped" : "";    //camera prediction plane

     useEffect(() => {
        const channel = new BroadcastChannel("debug");

        channel.onmessage = (event) => {
            const data = event.data;

            switch (data.type) {
                case "UPDATE_CONFIG":
                    console.log("--- update_config ---")
                    setShowEmotion(data.showEmotion);
                    setShowPerclos(data.showPerclos);
                    setMode(data.mode);
                    setMountpoints(data.mountpoints);
                    setInitializedMounts(false);
                    //send confirmation message to main window
                    channel.postMessage({ type: "RECEIVED_MESSAGE" });
                    break;

                case "STARTED_PROCESS":
                    console.log("--- started_process ---")
                    setHasStarted(data.hasStartedProcess ?? false);
                    break;
            }
        };

        return () => {
            setLoading(false);
            channel.close();
        };
    }, []);

    useEffect(() => {
        if (mountpoints.length > 0 && !initializedMounts){
            setStreamInfo({
                //streamId: 11, mid: 0,1,2,3
                "jetsonCamera": {streamID: mountpoints[0].id, mediaID: mountpoints[0].media[0].mid},
                "croppedFace": {streamID: mountpoints[0].id, mediaID: mountpoints[0].media[1].mid},
                "croppedEye1": {streamID: mountpoints[0].id, mediaID: mountpoints[0].media[2].mid},
                "croppedEye2": {streamID: mountpoints[0].id, mediaID: mountpoints[0].media[3].mid},
            });
            setInitializedMounts(true);
        }
    }, [mountpoints]);

    // console.log( hasStarted, showEmotion, showPerclos, mountpoints, loading)
    console.log("hasStarted:", hasStarted)
    console.log(">>", streamInfo)

    return (
        loading || mountpoints.length===0
        ? <div style={{display: "flex", justifyContent: "center", alignItems: "center", flex: 1}}>
            <span>
                <Text style={{marginRight: "15px", fontSize: "17px"}}>
                    {mountpoints.length===0 ? "Loading stream mountpoints" : ""}
                </Text>
                <Spin size="large" indicator={<LoadingOutlined />} /> 
            </span>
        </div>
        // : !hasStarted
        //     ? <div style={{display: "flex", justifyContent: "center", alignItems: "center", flex: 1}}>
        //         <span>
        //             <Text style={{marginRight: "10px", fontSize: "17px"}}>Waiting for {mode} to start</Text>
        //             <Spin indicator={<LoadingOutlined />} />
        //         </span>
        //     </div>
            : <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
                <Row style={{ height: "60vh"}}>
                    <Col 
                        span={14} 
                        style={{
                            backgroundColor: "#bbb8b8",     //"#ddd5c9"
                            height: "100%", overflow: "hidden", 
                            display: "flex", justifyContent: "center", alignItems: "center", 
                            borderRightStyle: "solid",
                            borderColor: "#c4c1c1"
                        }}
                    >
                        {/* jetson camera */}
                        {streamInfo["jetsonCamera"]?.streamID && <VideoPlayer streamID={streamInfo["jetsonCamera"].streamID!} mediaID={streamInfo["jetsonCamera"].mediaID!} />}
                    </Col>
                    <Col span={10}>
                        <Row style={{ height: '30vh'}}>
                            <Col 
                                span={24} 
                                style={{
                                    backgroundColor: "#bbb8b8",     //"#cbcec3"
                                    height: "100%", overflow: "hidden", 
                                    display: "flex", justifyContent: "center", alignItems: "center",
                                    borderBottomStyle: "solid",
                                    borderColor: "#c4c1c1"
                                }}
                            >
                                {/* cropped face */}
                                {streamInfo["croppedFace"]?.streamID && <VideoPlayer streamID={streamInfo["croppedFace"].streamID!} mediaID={streamInfo["croppedFace"].mediaID!} />}
                            </Col>
                        </Row>
                        <Row style={{ height: '30vh' }}>
                            <Col 
                                span={12} 
                                style={{
                                    backgroundColor: "#bbb8b8",   //"#b5d1c7"
                                    height: "100%", overflow: "hidden", 
                                    display: "flex", justifyContent: "center", alignItems: "center",
                                    borderRightStyle: "solid",
                                    borderColor: "#c4c1c1"
                                }}
                            >
                                {/* cropped eye 1 */}
                                {streamInfo["croppedEye1"]?.streamID && <VideoPlayer streamID={streamInfo["croppedEye1"].streamID!} mediaID={streamInfo["croppedEye1"].mediaID!} />}
                            </Col>
                            <Col 
                                span={12} 
                                style={{
                                    backgroundColor: "#bbb8b8",   //"#c6d5d8" 
                                    height: "100%", overflow: "hidden", 
                                    display: "flex", justifyContent: "center", alignItems: "center",
                                }}
                            >
                                {/* cropped eye 2 */}
                                {streamInfo["croppedEye2"]?.streamID && <VideoPlayer streamID={streamInfo["croppedEye2"].streamID!} mediaID={streamInfo["croppedEye2"].mediaID!} />}
                            </Col>
                        </Row>
                    </Col>
                </Row>
                <Row style={{ height: "40vh" }}>
                    <Col 
                        span={showEmotion || showPerclos ? 8 : 12} 
                        style={{
                            backgroundColor: "#bbb8b8",     //#bec5d3
                            display: "flex", justifyContent: "center", alignItems: "center", 
                            height: "100%",
                            padding: "2px"
                        }}
                    >
                        {/* camera pred plane */}
                        <CameraPredPlane 
                            rosTopic={cameraTopic}
                            isCalibrationMode={mode==="calibration"}
                        />
                    </Col>
                    {(showEmotion || showPerclos) && 
                        <Col 
                            span={7} 
                            style={{backgroundColor: "#dacfd9" }}
                        >
                            <Row style={{ height: "100%", display: "flex"}}>
                                {showEmotion && <Col span={24} style={{display: "flex", justifyContent: "center", alignItems: "center"}} >
                                    emotion
                                </Col>}
                                {showPerclos && <Col span={24} style={{display: "flex", justifyContent: "center", alignItems: "center"}}>
                                    perclos
                                </Col>}
                            </Row>
                        </Col>
                    }
                    <Col 
                        span={showEmotion || showPerclos ? 9 : 12} 
                        style={{
                            backgroundColor: "#d4c8c0", 
                            display: "flex", justifyContent: "center", alignItems: "center" 
                        }}
                    >
                        logs & metrics
                    </Col>
                </Row>
            </div>
    );
};

export default DebugScreen;