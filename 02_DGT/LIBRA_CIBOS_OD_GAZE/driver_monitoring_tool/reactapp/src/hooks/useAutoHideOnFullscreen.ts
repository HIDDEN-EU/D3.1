import { useEffect, useRef, useState } from "react";

export function useAutoHideOnFullscreen(
    containerRef: React.RefObject<HTMLElement | null>,
    isFullscreen: boolean,
    paused: boolean = false,
    hideDelayMs: number = 2000
) {
    const [visible, setVisible] = useState(true);
    const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const wasPausedRef = useRef<boolean>(false);
    const ignoreUntilRef = useRef<number>(0);

    useEffect(() => {
        const el = containerRef.current;
        if (!el || !isFullscreen) {
            wasPausedRef.current = false;
            setVisible(true); // always show normal cursor/no banner logic outside fullscreen
            return;
        }

        //keep banner hidden while paused
        if (paused) {
            if (timerRef.current) clearTimeout(timerRef.current);
            wasPausedRef.current = true;
            setVisible(false);
            return;
        }

        const showAndResetTimer = () => {
            //if it was paused earlier, ignore any mouse movements for a few ms
            if (Date.now() < ignoreUntilRef.current) return;
            //else set banner visible for 2s
            setVisible(true);
            if (timerRef.current) clearTimeout(timerRef.current);
            timerRef.current = setTimeout(() => setVisible(false), hideDelayMs);
        };

        if (wasPausedRef.current) {
            wasPausedRef.current = false;
            ignoreUntilRef.current = Date.now() + 1000;
        } else {
            showAndResetTimer(); //show immediately on entering fullscreen
        }
        
        el.addEventListener("mousemove", showAndResetTimer);

        return () => {
            el.removeEventListener("mousemove", showAndResetTimer);
            if (timerRef.current) clearTimeout(timerRef.current);
        };
    }, [containerRef, isFullscreen, hideDelayMs, paused]);

    return visible;
}
