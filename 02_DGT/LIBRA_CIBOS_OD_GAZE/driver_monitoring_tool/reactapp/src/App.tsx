import React, { useMemo, useEffect, useState } from 'react';
import { ConfigProvider, theme } from 'antd';
import { RouterProvider } from 'react-router-dom';
import { router } from './routes';
import { getConfig } from './helpers/config';
import { ThemeProvider, useTheme } from './context/ThemeContext';
import { useTranslation } from 'react-i18next';
import ErrorBoundary from './components/common/ErrorBoundary';
import './App.css';

// Day.js for Ant Design 5+
import dayjs from 'dayjs';
import 'dayjs/locale/el';
import 'dayjs/locale/en';

// Ant Design Locales
import enUS from 'antd/locale/en_US';
import elGR from 'antd/locale/el_GR';
import { JanusConfigProvider } from './context/JanusContext';
import { MountpointsProvider } from './context/MountpointsContext';
import { RosProvider } from './context/RosContext';

const AppConfigured: React.FC = () => {
  const config = getConfig();
  const { isDarkMode } = useTheme();
  const { i18n } = useTranslation();
  const [janusConfig, setJanusConfig] = useState<{janusUrl: string} | null>(null);

  // Sync dayjs locale whenever i18n language changes
  useEffect(() => {
    const lang = i18n.language.split('-')[0];
    dayjs.locale(lang);
  }, [i18n.language]);

  // Map i18next language to Ant Design locale
  const antLocale = useMemo(() => {
    switch (i18n.language.split('-')[0]) {
      case 'el':
        return elGR;
      case 'en':
      default:
        return enUS;
    }
  }, [i18n.language]);

  useEffect(() => {
    const conf = {
      janusUrl: config.janusUrl
    };
    setJanusConfig(conf);
  }, []);

  return (
    <ConfigProvider
      locale={antLocale}
      theme={{
        algorithm: isDarkMode ? theme.darkAlgorithm : theme.defaultAlgorithm,
        token: {
          colorPrimary: config.themeColor || '#E4007C',
          colorLink: '#E4007C',
          colorInfo: '#022C7E',
          borderRadius: 6,
          colorBgLayout: isDarkMode ? '#0A1A33' : '#F1F3F9',
        },
        components: {
          Layout: {
            siderBg: '#022C7E',
            headerBg: isDarkMode ? '#141414' : '#ffffff',
            headerHeight: 72,
          },
          Menu: {
            darkItemBg: 'transparent',
            darkItemColor: 'rgba(255, 255, 255, 0.75)',
            darkItemHoverColor: '#ffffff',
            darkItemHoverBg: 'rgba(255, 255, 255, 0.08)',
            darkItemSelectedBg: '#E4007C',
            darkItemSelectedColor: '#ffffff',
            itemBorderRadius: 10,
            itemMarginInline: 14,
            itemHeight: 46,
          },
        },
      }}
    >
      <JanusConfigProvider config={janusConfig}>
        <MountpointsProvider>
          <RosProvider configUrl={config.rosbridgeUrl}>
            <ErrorBoundary>
              <RouterProvider router={router} />
            </ErrorBoundary>
          </RosProvider>
        </MountpointsProvider>
      </JanusConfigProvider>
    </ConfigProvider>
  );
};

const App: React.FC = () => {
  return (
    <ThemeProvider>
      <AppConfigured />
    </ThemeProvider>
  );
};

export default App;
