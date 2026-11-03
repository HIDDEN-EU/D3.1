import React, { useCallback, useEffect, useRef, useState } from "react";
import { useJanusSession } from "../../context/JanusContext";
import type { PluginHandle } from "../../types/janusRelatedTypes";
import { WarningOutlined } from "@ant-design/icons";
import { Tooltip } from "antd";

interface JanusVideoPlayerProps {
    streamID: number;
    mediaID: string;
}

const statusLabel: Record<string, string> = {
    initializing: "Initializing...",
    connecting:   "Connecting to server...",
    preparing:    "Preparing stream...",
    starting:     "Starting stream...",
    playing:      "",
    error:        "Stream error",
    stalled:      "Stream stalled",
    disconnected: "Stream disconnected",
};

const statusColor: Record<string, string> = {
    initializing: "#555",
    connecting:   "#555",   //#aaa
    preparing:    "#555",   //#aaa
    starting:     "#555",   //#aaa
    playing:      "#555",   //#aaa
    error:        "#ff4d4f",
    stalled:      "#faad14",
    disconnected: "#ff4d4f",
};

const JanusVideoPlayer: React.FC<JanusVideoPlayerProps> = ({ streamID, mediaID }) => {
    const janusSession = useJanusSession();
    const janus = janusSession?.janus;
    const janusState = janusSession?.state;
    // const janus = useJanusSession();
    // const janusState = "";
    const streamingRef = useRef<PluginHandle | null>(null);
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const pendingStreamRef = useRef<MediaStream | null>(null);
    const opaqueId = useRef("streaming-" + Math.random().toString(36).substring(2));
    const [streamStatus, setStreamStatus] = useState<
        "initializing" | "connecting" | "preparing" | "starting" | "playing" | "error" | "stalled" | "disconnected"
    >("initializing");

    const videoRefCallback = useCallback((element: HTMLVideoElement | null) => {
        videoRef.current = element;
        if (element && pendingStreamRef.current) {
            element.srcObject = pendingStreamRef.current;
            element.play().catch(console.error);
        }
    }, []);

    useEffect(() => {
        if (!janus) {
            setStreamStatus("initializing");
            return;
        }
        setStreamStatus("connecting");
        if (videoRef.current) videoRef.current.srcObject = null;
        pendingStreamRef.current = null;
        streamingRef.current = null;
        console.log("--", streamID, mediaID)

        let cancelled = false;  //flag for handling overlapping requests

        janus.attach({
            plugin: "janus.plugin.streaming",
            opaqueId: opaqueId.current,

            success: (pluginHandle: PluginHandle) => {
                if (cancelled) {
                    pluginHandle.send?.({ message: {request: "stop"}});
                    return ;
                }
                streamingRef.current = pluginHandle;
                pluginHandle.send({
                    message: { request: "watch", id: streamID, media: [mediaID] },
                });
            },

            error: (e: unknown) => {
                // plugin attach failed or connection dropped
                console.error("Plugin error:", e);
                setStreamStatus("disconnected");
            },

            onmessage: (msg: Record<string, unknown>, jsep: unknown) => {
                if (cancelled) return ;
                if (msg["error"] || msg["message"] === "Request timed out") {
                    console.error("Error Streaming", msg["error"] || msg["message"]);
                    setStreamStatus("error");
                    return;
                }

                const result = msg["result"] as Record<string, unknown> | undefined;
                console.log(streamID, mediaID, result)
                if (result?.status === "preparing") setStreamStatus("preparing");
                if (result?.status === "starting") setStreamStatus("starting");
                if (result?.status === "started") setStreamStatus("starting"); // still buffering

                if (jsep && streamingRef.current) {
                    console.log("Processing WebRTC session...");
                    streamingRef.current.createAnswer({
                        jsep,
                        video: { recv: true },
                        success: (jsep: unknown) => {
                            if (cancelled) return;
                            console.log("WebRTC answer created, sending start request.");
                            streamingRef.current?.send({
                                message: { request: "start" },
                                jsep,
                            });
                        },
                        error: (e: unknown) => {
                            if (!cancelled){
                                console.error("WebRTC createAnswer error:", e);
                                setStreamStatus("error");
                            }
                        },
                    });
                }
            },

            onremotetrack: (track: MediaStreamTrack, mid: string, on: boolean) => {
                if (cancelled) return ;
                console.log(`Received track (${streamID}):`, { mid, mediaID, on, track, });
                // track removed — stream died mid-play
                if (mid === mediaID && !on) {
                    setStreamStatus("disconnected");
                    if (videoRef.current) videoRef.current.srcObject = null;
                    pendingStreamRef.current = null;
                    return;
                }
                //check for stale/leftover srcObject that might block attaching current track
                const alreadyAttached = (videoRef.current?.srcObject as MediaStream | null)
                    ?.getTracks().some((t) => t.id === track.id);
                if (mid !== mediaID || !on || alreadyAttached){ 
                    console.log(streamID,mid,mediaID, mid !== mediaID ,on, !on ,videoRef.current?.srcObject)
                    console.log(streamID, mediaID,"Skipping track attachment due to mismatched mid or track state");
                    return;
                }
                console.log("Attaching media stream to video element.", streamID, mediaID);
                const stream = new MediaStream([track]);
                pendingStreamRef.current = stream;

                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    videoRef.current.play()
                    //handle race condition - abort first play call if a second comes
                    .catch((e) => {
                        if (cancelled) return;
                        if (e.name === "AbortError" && videoRef.current) {
                            videoRef.current!.onloadedmetadata = () => {
                                console.log("metadata loaded, playing...");
                                videoRef.current?.play()
                                    .catch((e) => {
                                        if (cancelled) return;
                                        console.error(e);
                                        setStreamStatus("error");
                                    });
                            };
                        }
                    });
                }
            },

            oncleanup: () => {
                if (cancelled) return ;
                console.warn("Plugin cleanup");
                streamingRef.current = null;
                pendingStreamRef.current = null;
                if (videoRef.current) videoRef.current.srcObject = null;
                setStreamStatus("disconnected");
            },

            ondetached: () => {
                // fires when Janus explicitly detaches the plugin
                if (cancelled) return ;
                console.warn("Plugin detached");
                setStreamStatus("disconnected");
            },
        });

        return () => {
            cancelled = true;
            if (streamingRef.current) {
                streamingRef.current.send({ message: { request: "stop" } });
                streamingRef.current = null;
            }
        };
    }, [janus, streamID, mediaID]);

    // console.log("--janusState:", janusState);
    return (
        <div style={{ position: "relative", width: "100%", height: "100%", background: "transparent" }}>
            {janusState && janusState?.trim()!=="" && 
                <Tooltip title={janusState} placement="rightBottom" getPopupContainer={() => document.body}>
                    <span style={{display: "inline-block", verticalAlign: 'middle', cursor: "pointer"}}>
                        <WarningOutlined style={{color: "#e9a455", fontSize: "20px", padding: "5px"}} />
                    </span>
            </Tooltip>}
            {streamStatus !== "playing" && (
                <div style={{
                    position: "absolute",
                    inset: 0,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    background: "transparent",
                    zIndex: 10,
                    gap: "12px",
                    pointerEvents: "none",
                }}>
                    {streamStatus === "error" || streamStatus === "disconnected" ? (
                        <span style={{ fontSize: "24px" }}>
                            {streamStatus === "disconnected" ? "🔌" : "⚠️"}
                        </span>
                    ) : (
                        <div style={{
                            width: "40px",
                            height: "40px",
                            border: "3px solid #444",
                            borderTop: `3px solid #777575`,    //${statusColor[streamStatus]}
                            borderRadius: "50%",
                            animation: "spin 0.8s linear infinite",
                        }} />
                    )}
                    <span style={{ color: statusColor[streamStatus], fontSize: "13px" }}>
                        {statusLabel[streamStatus]}
                    </span>
                </div>
            )}
            <video
                ref={videoRefCallback}
                onPlaying={() => setStreamStatus("playing")}
                onWaiting={() => setStreamStatus("starting")}
                onStalled={() => setStreamStatus("stalled")}
                onEnded={() => setStreamStatus("disconnected")}
                onError={() => setStreamStatus("disconnected")} 
                style={{
                    width: "100%",
                    height: "100%",
                    display: "block",
                    objectFit: "contain",
                    opacity: streamStatus === "playing" ? 1 : 0,
                }}
                muted
                playsInline
                disablePictureInPicture
            />
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
    );
};

export default React.memo(JanusVideoPlayer);