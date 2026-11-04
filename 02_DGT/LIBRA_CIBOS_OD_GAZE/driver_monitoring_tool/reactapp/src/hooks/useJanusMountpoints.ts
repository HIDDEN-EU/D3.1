import { useEffect, useState } from "react";
import { useJanusSession } from "../context/JanusContext";

interface JanusMedia {
  mid: string;
  type: "video" | "audio" | "data";
  codec: string;
}

export interface JanusMountpoint {
  id: number;
  description: string;
  type: string;
  media: JanusMedia[];
}

export interface PluginHandle {
  send: (options: { message: Record<string, unknown>; jsep?: unknown }) => void;
  createAnswer: (options: {
    jsep: unknown;
    video?: { recv?: boolean; add?: boolean };
    media?: { audioSend: boolean; videoSend: boolean };
    success: (jsep: unknown) => void;
    error: (e: unknown) => void;
  }) => void;
}

export const useJanusMountpoints = () => {
  const janus = useJanusSession()?.janus;
  const [mountpoints, setMountpoints] = useState<JanusMountpoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!janus) return;

    let handle: any = null;
    janus.attach({
      plugin: "janus.plugin.streaming",
      opaqueId: "list-" + Math.random().toString(36).substring(2),

      success: (pluginHandle: PluginHandle) => {
        handle = pluginHandle;
        handle.send({
          message: { request: "list" },
          success: (result: any) => {
            const list: JanusMountpoint[] = result?.list ?? [];
            setMountpoints(list);
            setLoading(false);
            // detach — we only needed this handle for the list
            handle.detach();
          }
        });
      },

      onmessage: (msg: any) => {
        // list response also comes through onmessage in some Janus versions
        if (msg?.streaming === "list") {
          setMountpoints(msg.list ?? []);
          setLoading(false);
          handle?.detach();
        }
        if (msg?.error) {
          console.log('error-msg-', msg.error);
          setError(msg.error);
          setLoading(false);
        }
      },

      error: (e: string) => {
        console.log('error-', e)
        setError(e);
        setLoading(false);
      },

      oncleanup: () => {
        handle = null;
      },
    });

    return () => {
      handle?.detach();
    };
  }, [janus]);

  return { mountpoints, loading, error };
};