import { useCallback, useEffect, useRef } from "react";
import { useTheme } from "../../../context/ThemeContext";
import { useRosTopic } from "../../../hooks/useRosTopic";

const generateGridTicks = ([min, max]: number[]) => {
    const ticks = [];
    for (let i = Math.ceil(min); i <= Math.floor(max); i++) {
        ticks.push(i);
    }
    return ticks;
};

const CALIBRATION_DOMAIN = {
    x: [-1.1, 1.1],
    y: [-1.1, 1.1]
};

const INFERENCE_DOMAIN = {
    x: [-0.1, 12.1],
    y: [-0.1, 6.1]
};

const CALIBRATION_TICKS = {
    x: generateGridTicks(CALIBRATION_DOMAIN.x),
    y: generateGridTicks(CALIBRATION_DOMAIN.y)
};

const INFERENCE_TICKS = {
    x: generateGridTicks(INFERENCE_DOMAIN.x),
    y: generateGridTicks(INFERENCE_DOMAIN.y)
};


interface CameraPredPlaneProps{
    rosTopic: string;
    isCalibrationMode: boolean;     //calibration or inference mode
};

const CameraPredPlane: React.FC<CameraPredPlaneProps> = ({
    rosTopic="",
    isCalibrationMode
}) => {
    const { isDarkMode } = useTheme();
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const latestPointRef = useRef<{x:number; y:number;} | null>(null);

    const xAxisDomain = isCalibrationMode ? CALIBRATION_DOMAIN.x : INFERENCE_DOMAIN.x ;
    const yAxisDomain = isCalibrationMode ? CALIBRATION_DOMAIN.y : INFERENCE_DOMAIN.y ;
    const xAxisTicks = isCalibrationMode ? CALIBRATION_TICKS.x : INFERENCE_TICKS.x ;
    const yAxisTicks = isCalibrationMode ? CALIBRATION_TICKS.y : INFERENCE_TICKS.y ;

    const cameraPredictions = useRosTopic<{point: {x:number; y:number;}}>(rosTopic, "geometry_msgs/PointStamped", rosTopic!=='');
    // console.log(`preds[${rosTopic}]:`, cameraPredictions,)

    //keep latest point from camera predictions (within grid limits)
    useEffect(() => {
        const p = cameraPredictions?.point;
        if (p && checkPointLimits(p)){
            latestPointRef.current = p;
        } else {
            latestPointRef.current = null;
        }
    }, [cameraPredictions, xAxisDomain, yAxisDomain]);

    //display only points inside grid
    const checkPointLimits = (p: {x:number, y:number}) => {
        return (p.x>=xAxisDomain[0] && p.x<=xAxisDomain[1] && p.y>=yAxisDomain[0] && p.y<=yAxisDomain[1]);
    };

    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return ;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        const { width, height } = canvas.getBoundingClientRect();   //get actual on-screen CSS pixel size
        if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
            canvas.width = width * dpr;
            canvas.height = height * dpr;
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const bg = isDarkMode ? "#000000" : "#ffffff";
        const gridColor = isDarkMode ? "#333333" : "#e0e0e0";
        const pointColor = "#8884d8";

        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, width, height);

        //domain -> pixel mapping
        const mapX = (x: number) => ((x - xAxisDomain[0]) / (xAxisDomain[1] - xAxisDomain[0])) * width;
        const mapY = (y: number) => ((y - yAxisDomain[0]) / (yAxisDomain[1] - yAxisDomain[0])) * height;

        //grid lines
        ctx.strokeStyle = gridColor;
        ctx.lineWidth = 1;
        xAxisTicks.forEach((tick) => {
            const px = mapX(tick);
            ctx.beginPath();
            ctx.moveTo(px, 0);
            ctx.lineTo(px, height);
            ctx.stroke();
        });
        yAxisTicks.forEach((tick) => {
            const py = mapY(tick);
            ctx.beginPath();
            ctx.moveTo(0, py);
            ctx.lineTo(width, py);
            ctx.stroke();
        });

        //draw prediction point
        const p = latestPointRef.current;
        if (p) {
            ctx.fillStyle = pointColor;
            ctx.beginPath();
            ctx.arc(mapX(p.x), mapY(p.y), 8, 0, Math.PI * 2);
            ctx.fill();
        }
    }, [isDarkMode, xAxisDomain, yAxisDomain, xAxisTicks, yAxisTicks]);

    //rAF loop caps redraws at the display refresh rate regardless of ROS message rate 
    useEffect(() => {
        let frameId: number;
        const loop = () => {
            draw();
            frameId = requestAnimationFrame(loop);
        };
        frameId = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(frameId);
    }, [draw]);

    //redraw immediately on resize
    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;
        const ro = new ResizeObserver(() => draw());
        ro.observe(el);
        return () => ro.disconnect();
    }, [draw]);

    return (
        <div ref={containerRef} style={{width: "100%", height: "100%"}} >
            <canvas 
                ref={canvasRef} 
                style={{width: "100%", height: "100%", display: "block"}} 
            />
        </div>
    );
};

export default CameraPredPlane;