import { useEffect, useState } from "react";
import * as ROSLIB from "roslib";
import { useRos } from "../context/RosContext";

export function useRosTopic<T = unknown>(topicName: string, messageType: string, enabled: boolean = true) {
    const { ros } = useRos();
    const [message, setMessage] = useState<T | null>(null);

    useEffect(() => {
        if (!ros || !topicName || !enabled) return;

        const topic = new ROSLIB.Topic({
            ros,
            name: topicName,
            messageType,
        });

        const handler = (msg: T) => setMessage(msg);
        topic.subscribe(handler);

        return () => {
            topic.unsubscribe(handler);
        };
    }, [ros, topicName, messageType, enabled]);

    return message;
}