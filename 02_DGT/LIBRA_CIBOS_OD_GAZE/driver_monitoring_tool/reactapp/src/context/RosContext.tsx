import { createContext, useContext, useEffect, useRef, useState } from "react";
import * as ROSLIB from "roslib";

interface RosContextValue {
    ros: ROSLIB.Ros | null;
    connected: boolean;
}

const RosContext = createContext<RosContextValue | null>(null);

export const RosProvider = ({
    configUrl,
    children,
}: {
    configUrl: string;
    children: React.ReactNode;
}) => {
    const rosRef = useRef<ROSLIB.Ros | null>(null);
    const reconnectTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
    const [connected, setConnected] = useState(false);
    const [ros, setRos] = useState<ROSLIB.Ros | null>(null);

    useEffect(() => {
        if (!configUrl) return;

        const connect = () => {
            const rosInstance = new ROSLIB.Ros({ url: configUrl });

            rosInstance.on("connection", () => {
                console.log("Connected to rosbridge");
                setConnected(true);
                setRos(rosInstance);
            });

            rosInstance.on("error", (e) => {
                console.error("rosbridge error:", e);
                setConnected(false);
            });

            rosInstance.on("close", () => {
                console.warn("rosbridge connection closed");
                setConnected(false);
                setRos(null);
                reconnectTimeout.current = setTimeout(connect, 5000);
            });

            rosRef.current = rosInstance;
        };

        connect();

        return () => {
            if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
            rosRef.current?.close();
            rosRef.current = null;
        };
    }, [configUrl]);

    return (
        <RosContext.Provider value={{ ros, connected }}>
            {children}
        </RosContext.Provider>
    );
};

export const useRos = (): RosContextValue => {
    const ctx = useContext(RosContext);
    if (!ctx) throw new Error("useRos must be used within a RosProvider");
    return ctx;
};