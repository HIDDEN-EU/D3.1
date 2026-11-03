import React, { useEffect, useRef, useState, } from 'react';
import { Button, Card, Col, Row, Typography, message, Switch } from 'antd';
import { AimOutlined, ExportOutlined, EyeOutlined, PlayCircleFilled } from '@ant-design/icons';
import { dataManager } from '../helpers/dataManager';
import '../components/gazeTracking/chart.css';
import '../components/gazeTracking/debugScreen.css';
import Chart from '../components/gazeTracking/Chart';
import { useJanusMountpoints } from '../context/MountpointsContext';
import { useTheme } from '../context/ThemeContext';
import '../components/videoPlayer/videoPlayer.css';
import FullScreenVideo from '../components/videoPlayer/FullScreenVideo';
import './gazeTrackingPage.css';
import ConfirmationModal from '../components/common/ConfirmationModal';

const { Title } = Typography;

const GazeTrackingPage: React.FC = () => {
  const { isDarkMode } = useTheme();
  const [fullscreenElem, setFullscreenElem] = useState<Element | null>(null);
  const isFullscreen = !!fullscreenElem;
  const chartRef = useRef(null);
  const [isInitializingCalibration, setIsInitializingCalibration] = useState<boolean>(false);
  const [isInitializingGazePointMap, setIsInitializingGazePointMap] = useState<boolean>(false);
  const [trueData, setTrueData] = useState<{x:number, y:number}[]>([]); //initial true gaze points
  const [isCalibration, setIsCalibration] = useState<boolean>(true);    //calibration or gazePointMapping mode

  const [displayData, setDisplayData] = useState<{x:number; y:number;}[]>([]);  //for displaying true points on screen
  const [index, setIndex] = useState(0);
  const [displayPredictedData, setDisplayPredictedData] = useState<{x:number; y:number;}[]>([]);  //for displaying predicted points on screen
  const [openActionModal, setOpenActionModal] = useState<boolean>(false);   //for calibration: modal with available actions after each point
  const [modalProps, setModalProps] = useState<{
    onRetake: (() => void) | undefined,     //retake same point
    onMove: (() => void) | undefined,       //move to next point
    onSave: (() => Promise<void>) | undefined,  //save process > calculate homography matrix
    onCancel: (() => void) | undefined,     //cancel process
  }>({
    onRetake: undefined,
    onMove: undefined,
    onSave: undefined,
    onCancel: undefined,
  });
  const hasShownFirstPointRef = useRef<boolean>(false); //keep ref to avoid extra delay when retaking 1st point

  const [showEmotion, setShowEmotion] = useState<boolean>(false);
  const [showPerclos, setShowPerclos] = useState<boolean>(false);
  const [videoMode, setVideoMode] = useState<boolean>(false);
  const [roadFacingMode, setRoadFacingMode] = useState<boolean>(false);
  const debugRef = useRef<BroadcastChannel | null>(null);
  const videoRoadRef = useRef(null);
  const videoExampleRef = useRef(null);
  const { mountpoints, loading } = useJanusMountpoints();
  const [streamInfo, setStreamInfo] = useState<Record<string, {streamID?: number; mediaID?: string}>>({
    "roadVideo": {streamID: undefined, mediaID: undefined},
    "videoExample": {streamID: undefined, mediaID: undefined}
  });
  //--- refs for broadcasting updates to debug screen
  const debugConfigRef = useRef(false);
  const retryIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const debugModeRef = useRef('');

  console.log("mountpoints:", mountpoints);

    const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    const clearData = () => {
        setIndex(0);
        setTrueData([]);
        setDisplayData([]);
        setDisplayPredictedData([]);
        hasShownFirstPointRef.current = false;
    };

    const handleCancelProcess = async () => {
        //cancel button is clicked --> mark exit so listener won't cancel again
        //  (on esc, fullscreen is already closed)
        if (document.fullscreenElement) {
            intentionalExitRef.current = true;
            document.exitFullscreen();
        }
        setOpenActionModal(false);
        const hideLoading = message.loading("Cancelling process...", 0);
        try {
            console.log('...terminating process');
            clearData();
            await dataManager.post('/demo/stop', {}, {skipGlobalError: true});
            hideLoading();
            message.success("Process was cancelled", 3);
        } catch (error){
            console.error(error);
            hideLoading();
            message.error("An error occured when cancelling process.", 3);
        }
    };

    const intentionalExitRef = useRef<boolean>(false);    //either when process is completed or stop button is clicked
    const cancelFunctionRef = useRef(handleCancelProcess);  //fullscreen listener is registered once --> read latest handleCancelProcess from ref
    cancelFunctionRef.current = handleCancelProcess;

    const toggleFullscreen = (mode: "calibration" | "gazeMapping") => {
        const isVideo = (videoMode || roadFacingMode);
        const elem = mode==="gazeMapping"
            ? isVideo ? (videoMode ? videoExampleRef.current : videoRoadRef.current) : chartRef.current
            : chartRef.current;
        if (!document.fullscreenElement) {
            //enter fullscreen
            if (elem.requestFullscreen) {
                elem.requestFullscreen();
            } else if (elem.webkitRequestFullscreen) { //Safari
                elem.webkitRequestFullscreen();
            }
        } else {
            //exit fullscreen
            if (document.exitFullscreen) {
                intentionalExitRef.current = true;  //process was completed - not canceled
                document.exitFullscreen();
            }
        }
    };

    //track fullscreen element --> cancel process if user exits with esc
    // (browser handles esc itself, keydown is not fired)
    useEffect(() => {
        const onFullscreenChange = () => {
            const elem = document.fullscreenElement;
            setFullscreenElem(elem);

            if (!elem) {    //exited fullscreen
                if (intentionalExitRef.current) {
                    intentionalExitRef.current = false;  //exited from code --> reset flag
                } else {
                    cancelFunctionRef.current();  //exited by esc
                }
            }
        };

        document.addEventListener('fullscreenchange', onFullscreenChange);
        return () => document.removeEventListener('fullscreenchange', onFullscreenChange);
    }, []);

    //display only points inside grid
    const checkPointLimits = (p: {x:number, y:number}) => {
        return (p.x>=0 && p.x<=12 && p.y>=0 && p.y<=6);
    };

    useEffect(() => {
        if (trueData.length === 0 || !isFullscreen) return;
        if (openActionModal) return;   //pause timer while modal is open

        if (index < trueData.length){
            //wait for the escape message to disappear before showing first point
            const delay = index===0 && !hasShownFirstPointRef.current ? 4000 : 0;
            const showTimer = setTimeout(() => {
                hasShownFirstPointRef.current = true;
                if (checkPointLimits(trueData[index])) setDisplayData([trueData[index]]);
                console.log(trueData[index], index)
                handleChangePoint(trueData[index]); 
            }, delay);

            const endTimer = setTimeout(() => {
                //-gaze mapping mode --> continue to next point
                if (!isCalibration) {
                    setIndex((prev) => prev + 1);
                    return;
                }

                //-calibration mode
                handlePauseProcess();   //pause process to avoid invalid gazes while modal is open

                const isLast = index === trueData.length - 1;
                setModalProps({
                    onRetake: () => setOpenActionModal(false),
                    onMove: isLast ? undefined : () => {
                        setOpenActionModal(false);
                        setIndex((prev) => prev + 1);
                    },
                    onSave: isLast ? () => handleFinishCalibration() : undefined,
                    onCancel: isLast ? () => handleCancelProcess() : undefined
                });
                setOpenActionModal(true);
            }, delay + 7100);   //move to next point every ~7s

            return () => {
                clearTimeout(showTimer);
                clearTimeout(endTimer);
            };
        }

        //if all points are shown
        if (index === trueData.length && !isCalibration) {
            //- gaze point mapping mode
            setDisplayData([]);
            toggleFullscreen("gazeMapping");
            clearData();
            message.success("Gaze Point Mapping completed", 3);
        }

    }, [trueData, index, isFullscreen, openActionModal]);

    const handleStartCalibration = async () => {
        clearData();
        setIsInitializingCalibration(true);
        setIsCalibration(true);
        try {
            const response = await dataManager.post(
                '/points/reference/initialize', 
                {calibration: "True"}, 
                {skipGlobalError: true}
            );
            const points = response.ref_points.map(
                (p: {x_value: number, y_value: number}) => ({x: p.x_value, y: p.y_value}));
            console.log(points);
            setTrueData(points);
            alertDebugScreen(true);
            toggleFullscreen("calibration");
        } catch (error){
            console.error(error);
            message.error("An error occured, please try again.", 3);
            alertDebugScreen(false);
        } finally {
            setIsInitializingCalibration(false);
        }
    };

    const handleStartGazePointMapping = async () => {
        clearData();
        setIsInitializingGazePointMap(true);
        setIsCalibration(false);
        try {
            const response = await dataManager.post(
                '/points/reference/initialize', 
                {calibration: "False"}, 
                {skipGlobalError: true}
            );
            const points = response.ref_points.map(
                (p: {x_value: number, y_value: number}) => ({x: p.x_value, y: p.y_value}));
            console.log(points);
            setTrueData(points);
            alertDebugScreen(true);
            toggleFullscreen("gazeMapping");
        } catch (error){
            console.error(error);
            message.error("An error occured, please try again.", 3);
            alertDebugScreen(false);
        } finally {
            setIsInitializingGazePointMap(false);
        }
    };

    const handleChangePoint = async (point: {x: number, y: number}) => {
        try {
            console.log('...changing point');
            await dataManager.post(
                '/points/reference/update', 
                {x_value: point.x, y_value: point.y}, 
                {skipGlobalError: true}
            );

            if (!isCalibration) {
                //gazeMapping mode: wait 2s before getting prediction
                await sleep(2000);
                await getPredictedPoint();
            }
        } catch (error){
            console.error(error);
        }
    };

    const handleFinishCalibration = async () => {
        toggleFullscreen("calibration");
        setOpenActionModal(false);
        clearData();
        const hideLoading = message.loading("Finishing Calibration...", 0);
        try {
            console.log('...finishing');
            const response = await dataManager.post('/points/prediction/calculate', {}, {skipGlobalError: true});
            console.log(response);
            hideLoading();
            message.success("Calibration completed", 3);
        } catch (error){
            console.error(error);
            hideLoading();
            message.error("An error occured when completing calibration.", 3);
        }
    };

    const getPredictedPoint = async () => {
        try {
            console.log('..getting prediction');
            const response = await dataManager.get('/points/prediction', {skipGlobalError: true});
            const prediction = {x: response.x_value, y: response.y_value};
            console.log('--pred:', prediction)
            if (checkPointLimits(prediction)) {
                setDisplayPredictedData([prediction]);
            } else {
                setDisplayPredictedData([]);
            }
        } catch (error){
            console.error(error);
            // message.error("An error occured when getting prediction.", 3);
            setDisplayPredictedData([]);
        }
    };

    const handlePauseProcess = async () => {
        try {
            // console.log('...pausing');
            await dataManager.post('/demo/pause', {}, {skipGlobalError: true});
        } catch (error){
            console.error(error);
        }
    };


    //----- broadcasting to debug screen handling ---
    useEffect(() => {
        debugRef.current = new BroadcastChannel("debug");

        debugRef.current.onmessage = (event) => {
            console.log("From debug window:", event.data);
            if (event.data?.type === "RECEIVED_MESSAGE"){
                debugConfigRef.current = true;
                if (retryIntervalRef.current){
                    clearInterval(retryIntervalRef.current);
                    retryIntervalRef.current = null;
                }
            }
        };

        return () => {
            debugRef.current?.close();
            if (retryIntervalRef.current) clearInterval(retryIntervalRef.current);
        };
    }, []);

    //--if mountpoints change, update debug window
    useEffect(() => {
        if (debugConfigRef.current === true && debugModeRef.current!==''){
            console.log('--- repeat update_config');
            const config = {
                type: "UPDATE_CONFIG",
                showEmotion: debugModeRef.current === "calibration" ? showEmotion : true,
                showPerclos: debugModeRef.current === "calibration" ? showPerclos : true,
                mode: debugModeRef.current,
                mountpoints
            };
            debugRef.current?.postMessage(config);
        }
    }, [mountpoints]);

    const openDebugScreen = (mode: "calibration" | "inference") => {
        debugModeRef.current = mode;
        const width = window.screen.availWidth;
        const height = window.screen.availWidth;
        window.open(
            "/debug",
            "_blank",
            `width=${width},height=${height},resizable=yes,fullscreen=yes`
        );

        //--reset broadcasting refs
        debugConfigRef.current = false;
        if (retryIntervalRef.current) clearInterval(retryIntervalRef.current);

        const config = {
            type: "UPDATE_CONFIG",
            showEmotion: mode === "calibration" ? showEmotion : true,
            showPerclos: mode === "calibration" ? showPerclos : true,
            mode,
            mountpoints
        };

        let attempts = 0;
        const maxAttempts = 25; //~5s

        //wait for debug page to acknowledge config
        retryIntervalRef.current = setInterval(() => {
            //if debug screen has answered || reached max attempts
            if (debugConfigRef.current || attempts >= maxAttempts) {
                if (retryIntervalRef.current) clearInterval(retryIntervalRef.current);
                retryIntervalRef.current = null;
                if (!debugConfigRef.current) {
                    console.warn("Debug screen never acknowledged config — popup may be blocked or failed to load.");
                    // message.error("Could not connect to debug screen.", 3);
                }
                return;
            }
            attempts += 1;
            debugRef.current?.postMessage(config);
        }, 200);
    };

    const alertDebugScreen = (hasStarted: boolean) => {
        //wait for the debug page to initialize
        // setTimeout(() => {
            debugRef.current?.postMessage({
                type: "STARTED_PROCESS",
                hasStartedProcess: hasStarted,
            });
        // }, 100);
    };

    const handleInferenceMode = (value: boolean, mode: "video" | "road") => {
        if (mode === "video"){
            setVideoMode(value);
            setRoadFacingMode(false);
        } else {
            setRoadFacingMode(value);
            setVideoMode(false);
        }
    };

    useEffect(() => {
        if (mountpoints.length > 0){
            setStreamInfo({
                "roadVideo": {streamID: mountpoints[1].id, mediaID: mountpoints[1].media[0].mid},   //streamID: 12, mediaID: v
                "videoExample": {streamID: mountpoints[1].id, mediaID: mountpoints[1].media[0].mid}
            });
        }
    }, [mountpoints]);

    const handleStartInferenceVideo = () => {
        setIsCalibration(false);
        // while (loading) {console.log("waiting for mountpoints")};
        alertDebugScreen(true);
        toggleFullscreen("gazeMapping");
    };

  return (
    <>
        <div style={{ minHeight: '100%', display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {/* -- menu options (start calibration or gazePointMapping) -- */}
            <>
                <Row gutter={[24, 24]} justify="center">
                    <Col xs={24} md={20} lg={12} xl={11} style={{ display: 'flex' }}>
                        <Card 
                            className="gt-card gt-card--calibration"
                            title={
                                <div className="gt-card-title">
                                    <div className="gt-card-heading">
                                        <span className="gt-card-icon"><AimOutlined /></span>
                                        <Title level={4}>Calibration</Title>
                                    </div>
                                    <Button onClick={() => handleStartCalibration()} loading={isInitializingCalibration} disabled={isInitializingGazePointMap}
                                        className="gt-start-btn" size="large" icon={<PlayCircleFilled/>}
                                    >
                                        Start 
                                    </Button>
                                </div>
                            }
                        >
                            <div className="gt-section-label">Debug Screen Configuration</div>
                            <div className="gt-options">
                                <div className="gt-option">
                                    <label style={{color: isDarkMode ? "rgba(255, 255, 255, 0.85)" : "#303030"}}>Facial Emotion Estimation</label>
                                    <Switch
                                        className='calibration-switch'
                                        checked={showEmotion}
                                        onChange={setShowEmotion}
                                    />
                                </div>
                                <div className="gt-option">
                                    <label style={{color: isDarkMode ? "rgba(255, 255, 255, 0.85)" : "#303030"}}>PERCLOS</label>
                                    <Switch
                                        className='calibration-switch'
                                        checked={showPerclos}
                                        onChange={setShowPerclos}
                                    />
                                </div>
                            </div>
                            <div className="gt-card-footer">
                                <Button className="gt-debug-btn"
                                    icon={<ExportOutlined style={{fontSize: "15px"}}/>}
                                    onClick={() => openDebugScreen("calibration")}
                                >   
                                    Open Debug Screen
                                </Button>
                            </div>
                        </Card>
                    </Col>
                    <Col xs={24} md={20} lg={12} xl={11} style={{ display: 'flex' }}>
                        <Card 
                            className="gt-card gt-card--mapping"
                            title={
                                <div className="gt-card-title">
                                    <div className="gt-card-heading">
                                        <span className="gt-card-icon"><EyeOutlined /></span>
                                        <Title level={4}>Gaze Point Mapping</Title>
                                    </div>
                                    <Button onClick={() => (videoMode || roadFacingMode) ? handleStartInferenceVideo() : handleStartGazePointMapping() } 
                                        disabled={isInitializingCalibration}
                                        loading={/*(videoMode || roadFacingMode) ? loading :*/ isInitializingGazePointMap} 
                                        className="gt-start-btn" size="large" icon={<PlayCircleFilled/>}
                                    >
                                        Start
                                    </Button> 
                                </div>
                            } 
                        >
                            <div className="gt-section-label">Gaze Screen Configuration</div>
                            <div className="gt-options">
                                <div className="gt-option">
                                    <label style={{color: isDarkMode ? "rgba(255, 255, 255, 0.85)" : "#303030"}}>Video-example mode</label>
                                    <Switch
                                        className='inference-switch'
                                        checked={videoMode}
                                        onChange={(checked: boolean) => handleInferenceMode(checked, "video")}
                                    />
                                </div>
                                <div className="gt-option">
                                    <label style={{color: isDarkMode ? "rgba(255, 255, 255, 0.85)" : "#303030"}}>Road-facing mode</label>
                                    <Switch
                                        className='inference-switch'
                                        checked={roadFacingMode}
                                        onChange={(checked: boolean) => handleInferenceMode(checked, "road")}
                                    />
                                </div>
                            </div>
                            <div className="gt-card-footer">
                                <Button className="gt-debug-btn"
                                    icon={<ExportOutlined style={{fontSize: "15px"}}/>}
                                    onClick={() => openDebugScreen("inference")}
                                >   
                                    Open Debug Screen
                                </Button>
                            </div>
                        </Card>
                    </Col>
                </Row>
            </>
            {/* -- chart will be displayed only in fullscreen -- */}
            <Chart 
                chartRef={chartRef}
                isFullscreen={isFullscreen && fullscreenElem === chartRef.current} 
                displayData={displayData} 
                displayPredictedData={displayPredictedData} 
                isCalibrationMode={isCalibration} 
                isModalOpen={openActionModal}
                onCancel={handleCancelProcess}
            />
            {/* -- videos for gaze point mapping will be displayed only in fullscreen -- */}
            <FullScreenVideo
                videoRef={videoExampleRef}
                isFullscreen={isFullscreen && videoMode && fullscreenElem === videoExampleRef.current}
                streamID={streamInfo["videoExample"]?.streamID ?? 0}
                mediaID={streamInfo["videoExample"]?.mediaID ?? ''}
                isLoadingMountpoint={loading || mountpoints.length===0}
                onCancel={handleCancelProcess}
            />
            <FullScreenVideo
                videoRef={videoRoadRef}
                isFullscreen={isFullscreen && roadFacingMode && fullscreenElem === videoRoadRef.current}
                streamID={streamInfo["roadVideo"]?.streamID ?? 0}
                mediaID={streamInfo["roadVideo"]?.mediaID ?? ''}
                isLoadingMountpoint={loading || mountpoints.length===0}
                onCancel={handleCancelProcess}
            />
        </div>
        {/* -- options for calibration (fullscreen chart mode) -- */}
        <ConfirmationModal 
            open={openActionModal}
            getContainer={() => chartRef.current ?? document.body}
            {...modalProps}
        />
    </>
  );
};

export default GazeTrackingPage;
