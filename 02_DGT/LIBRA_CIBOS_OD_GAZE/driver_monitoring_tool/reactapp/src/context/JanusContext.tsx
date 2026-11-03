import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";

interface JanusInstance {
  attach: (options: Record<string, unknown>) => void;
  destroy: () => void;
}

declare global {
  interface Window {
    Janus: {
      init: (options: { debug: boolean | string; callback: () => void }) => void;
      isWebrtcSupported: () => boolean;
      attachMediaStream: (element: HTMLVideoElement, stream: MediaStream) => void;
      new (options: {
        server: string;
        iceServers: [];
        success: () => void;
        error: (e: string) => void;
        destroyed: () => void;
      }): JanusInstance;
    };
  }
}

interface Config {
  janusUrl: string;
}

const ConfigContext = createContext<Config | null>(null);
// const JanusContext = createContext<JanusInstance | null>(null);
const JanusContext = createContext<{janus: JanusInstance | null; state: string;} | null>(null);

export const JanusConfigProvider = ({
    config,
    children,
}: {
    config: Config | null;
    children: React.ReactNode;
}) => {
    const [janusSession, setJanusSession] = useState<JanusInstance | null>(null);
    const janusRef = useRef<JanusInstance | null>(null);
    const reconnectTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
    const isJanusInitialized = useRef(false);
    const [janusState, setJanusState] = useState<string>("");

    const initJanus = useCallback(() => {
        if (!config?.janusUrl) return;

        const Janus = window.Janus;

        const createSession = () => {
            janusRef.current = new Janus({
                server: config.janusUrl,
                iceServers: [],
                success: () => {
                    console.log("Janus session ready");
                    setJanusSession(janusRef.current);
                    setJanusState("");
                },
                error: (e) => {
                    console.error("Janus session error:", e);
                    setJanusSession(null);
                    setJanusState("Trying to reconnect to Janus server...");
                    console.log("Reconnecting after error...");
                    reconnectTimeout.current = setTimeout(createSession, 7000); //retry session only
                },
                destroyed: () => {
                    console.warn("Janus session destroyed");
                    janusRef.current = null;
                    setJanusSession(null);
                    setJanusState("Trying to reconnect to Janus server...");
                    console.log("Reconnecting after destroy...");
                    reconnectTimeout.current = setTimeout(createSession, 7000); //retry session only
                },
            });
        };

        if (isJanusInitialized.current) {
            // Janus.init already called — just create a new session
            createSession();
            return;
        }

        // first time — init then create session
        Janus.init({
            debug: false,
            callback: () => {
                if (!Janus.isWebrtcSupported()) {
                    console.error("No WebRTC support...");
                    return;
                }
                isJanusInitialized.current = true;
                createSession();
            },
        });
    }, [config]);

    useEffect(() => {
        initJanus();

        return () => {
            if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
            janusRef.current?.destroy();
            janusRef.current = null;
        };
    }, [initJanus]);

    return (
        <ConfigContext.Provider value={config}>
            <JanusContext.Provider /*value={janusSession}*/ value={{janus: janusSession, state: janusState}}>
                {children}
            </JanusContext.Provider>
        </ConfigContext.Provider>
    );
};

export const useJanusConfig = (): Config | null => useContext(ConfigContext);
// export const useJanusSession = (): JanusInstance | null => useContext(JanusContext);
export const useJanusSession = () : {janus: JanusInstance | null; state: string;} | null => useContext(JanusContext);

