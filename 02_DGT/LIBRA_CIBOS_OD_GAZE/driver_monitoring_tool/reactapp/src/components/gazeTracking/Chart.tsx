import { type Ref } from "react";
import { ResponsiveContainer, Scatter, ScatterChart, XAxis, YAxis } from "recharts";
import { useTheme } from "../../context/ThemeContext";
import { useAutoHideOnFullscreen } from "../../hooks/useAutoHideOnFullscreen";
import FullscreenHoverBanner from "./FullscreenHoverBanner";

const CirclePoint = (props) => {
  const { tooltipPosition, tooltipPayload, tooltipItem, isActive, payload, ...validSvgProps } = props;

  return (
    <circle 
      {...validSvgProps} 
      r={9}
    />
  );
};

interface ChartProps{
    chartRef: Ref<HTMLInputElement>;    //chart ref for entering fullscreen
    isFullscreen: boolean;
    displayData: {x:number; y:number;}[];   //data for true points
    displayPredictedData: {x:number; y:number;}[];  //data for predicted points
    isCalibrationMode: boolean;     //calibration or gazePointMapping mode
    isModalOpen: boolean;   //modal for showing available actions after each point
    onCancel: () => void; 
};

const Chart: React.FC<ChartProps> = ({
    chartRef,
    isFullscreen, 
    displayData=[], 
    displayPredictedData=[],
    isCalibrationMode,
    isModalOpen,
    onCancel
}) => {
    const { isDarkMode } = useTheme();
    const controlsVisible = useAutoHideOnFullscreen(chartRef as React.RefObject<HTMLElement>, isFullscreen, isModalOpen);

    return (
        <div
            ref={chartRef}
            className={isFullscreen ? "chart-wrapper" : ""}
            style={{
                cursor: isFullscreen && !controlsVisible && !isModalOpen ? "none" : "default",
                position: "relative",
                backgroundColor: isDarkMode ? "black" : "white",
            }}
        >
            {isFullscreen && <FullscreenHoverBanner visible={controlsVisible} onCancel={onCancel} />}
            <ResponsiveContainer 
                width="100%" height="100%" 
                key={isFullscreen ? 'fs' : 'normal'} 
                initialDimension={{ width: 700, height: 500 }}
            >
                <ScatterChart 
                    style={{ overflow: 'visible', cursor: 'inherit' }} 
                    responsive
                >
                    <XAxis type="number" dataKey="x" orientation='top' domain={[-0.05, 12.05]} hide />
                    <YAxis type="number" dataKey="y" reversed domain={[-0.05, 6.05]} hide />
                    <Scatter name="true" data={displayData} fill="#fa0404" shape={<CirclePoint/>} 
                        animationEasing='ease-out' animationDuration={400} />
                    {!isCalibrationMode && <Scatter name="predicted" data={displayPredictedData} fill="#8884d8" 
                        animationEasing='ease-in-out' animationDuration={300} shape={<CirclePoint/>} />}
                </ScatterChart>
            </ResponsiveContainer>
        </div>
    );
};

export default Chart;