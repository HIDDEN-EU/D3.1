import { createContext, useContext, useEffect, useRef, useState } from "react";
import { useJanusSession } from "./JanusContext";
import type { JanusMountpoint, PluginHandle } from "../types/janusRelatedTypes";

interface MountpointsContextValue {
  mountpoints: JanusMountpoint[];
  loading: boolean;
  error: string | null;
}

const MountpointsContext = createContext<MountpointsContextValue | null>(null);

export const MountpointsProvider = ({ children }: { children: React.ReactNode }) => {
  const janus = useJanusSession()?.janus;
  const [mountpoints, setMountpoints] = useState<JanusMountpoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const handleRef = useRef<PluginHandle | null>(null);

  useEffect(() => {
    if (!janus) return;

    setLoading(true);
    janus.attach({
      plugin: "janus.plugin.streaming",
      opaqueId: "list-" + Math.random().toString(36).substring(2),

      success: (pluginHandle: PluginHandle) => {
        handleRef.current = pluginHandle;
        pluginHandle.send({
          message: { request: "list" },
          success: (result: any) => {
            const list: JanusMountpoint[] = result?.list ?? [];
            setMountpoints(list);
            setLoading(false);
            //detach — we only needed this handle for the list
            handleRef.current?.detach();
          }
        });
      },

      onmessage: (msg: any) => {
        //list response also comes through onmessage in some Janus versions
        if (msg?.streaming === "list") {
          setMountpoints(msg.list ?? []);
          setLoading(false);
          handleRef.current?.detach();
        }
        if (msg?.error) {
          console.log('error-msg-', msg.error);
          setError(msg.error);
          setLoading(false);
        }
      },

      error: (e: string) => {
        console.log('error-', e);
        setError(e);
        setLoading(false);
      },

      oncleanup: () => {
        handleRef.current = null;
      },
    });

    return () => {
      handleRef.current?.detach();
    };
  }, [janus]); //refetch if Janus session reconnects

  return (
    <MountpointsContext.Provider value={{ mountpoints, loading, error }}>
      {children}
    </MountpointsContext.Provider>
  );
};

export const useJanusMountpoints = (): MountpointsContextValue => {
  const ctx = useContext(MountpointsContext);
  if (!ctx) throw new Error("useJanusMountpoints must be used within a MountpointsProvider");
  return ctx;
};