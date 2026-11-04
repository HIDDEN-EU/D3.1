import { createBrowserRouter } from 'react-router-dom';
import MainLayout from '../layout/MainLayout';
import GazeTrackingPage from '../pages/GazeTrackingPage';
import DebugScreen from '../components/gazeTracking/DebugScreen';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <MainLayout />,
    children: [
      {
        index: true,
        element: <GazeTrackingPage />,
      },
    ],
  },
  {
    path: '/debug',
    element: <DebugScreen />
  },
]);
