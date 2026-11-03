import React from 'react';
import { Layout, Typography, Menu, /*Avatar,*/ Dropdown, Switch, Space, theme } from 'antd';
import { 
  // SettingOutlined, 
  // FireFilled, 
  // UserOutlined, 
  // LogoutOutlined, 
  // ProfileOutlined, 
  BulbFilled, 
  BulbOutlined, 
  GlobalOutlined, 
  AimOutlined
} from '@ant-design/icons';
import type { MenuProps } from 'antd';
import { Outlet, useNavigate, /*useLocation*/ } from 'react-router-dom';
// import { dataManager } from '../helpers/dataManager';
import { useTheme } from '../context/ThemeContext';
import { useTranslation } from 'react-i18next';
import BrandGlyph from '../components/common/BrandGlyph';
import hiddenMark from '../assets/hidden_mark.png';
import '../App.css';
import ConnectionStatus from '../components/common/ConnectionStatus';

const { Header, Content, Footer, Sider } = Layout;
const { Text } = Typography;
const { useToken } = theme;

const MainLayout: React.FC = () => {
  const navigate = useNavigate();
  // const location = useLocation();
  const { isDarkMode, toggleTheme } = useTheme();
  const { t, i18n } = useTranslation();
  const { token } = useToken();

  const changeLanguage = (lng: string) => {
    i18n.changeLanguage(lng);
  };

  const languageItems: MenuProps['items'] = [
    { key: 'en', label: 'English', onClick: () => changeLanguage('en') },
    { key: 'el', label: 'Ελληνικά', onClick: () => changeLanguage('el') },
  ];

  // const profileMenu: MenuProps['items'] = [
  //   { key: 'profile', icon: <ProfileOutlined />, label: t('common.profile') },
  //   { type: 'divider' },
  //   { key: 'logout', icon: <LogoutOutlined />, label: t('common.logout'), onClick: () => dataManager.logout() },
  // ];

  // Map route to selected menu key
  let selectedKey = '1';
  // if (location.pathname.includes('/settings')) selectedKey = '2';

  const menuItems = [
    { key: '1', icon: <AimOutlined />, label: t('common.gaze_tracking'), onClick: () => navigate('/') },
    // { key: '2', icon: <SettingOutlined />, label: t('common.settings'), onClick: () => navigate('/settings') },
  ];

  const currentLangLabel = i18n.language.startsWith('el') ? 'EL' : 'EN';

  return (
    <Layout className="main-layout">
      {/* Top Navigation Bar */}
      <Header
        className="main-header"
        style={{
          background: token.colorBgContainer,
          boxShadow: '0 2px 12px rgba(2, 44, 126, 0.08)',
          '--brand-accent': token.colorPrimary,
        } as React.CSSProperties}
      >
        {/* Top Left: Logo pointing to Home */}
        <div className="logo-container" onClick={() => navigate('/')}>
          <img src={hiddenMark} alt="" className="logo-mark" />
          <span className="logo-wordmark" style={{ color: token.colorPrimary, borderLeftColor: token.colorBorder }}>HIDDEN</span>
        </div>
        
        {/* Top Right: Language, Theme Switch & Profile Menu */}
        <div
          className="header-actions"
          style={{ background: token.colorFillQuaternary, borderColor: token.colorBorderSecondary }}
        >
          <Space size="large">
            {/* <Dropdown menu={{ items: languageItems }} placement="bottomRight" arrow>
              <div className="profile-hover" style={{ cursor: 'pointer', padding: '4px 8px', borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <GlobalOutlined style={{ fontSize: '16px', color: token.colorText }} />
                <Text strong style={{ fontSize: '12px', color: token.colorText }}>{currentLangLabel}</Text>
              </div>
            </Dropdown> */}

            <Switch 
              checked={isDarkMode} 
              onChange={toggleTheme} 
              checkedChildren={<BulbFilled />} 
              unCheckedChildren={<BulbOutlined />} 
            />
          </Space>

          <span className="header-actions-divider" style={{ background: token.colorBorder }} />

          <span className="header-status">
            <ConnectionStatus />
          </span>

          {/* <Dropdown menu={{ items: profileMenu }} placement="bottomRight" arrow>
            <div className="profile-menu profile-hover">
              <Avatar icon={<UserOutlined />} src="https://api.dicebear.com/7.x/miniavs/svg?seed=7" className="profile-avatar" />
              <Text strong className="profile-name" style={{ color: token.colorText }}>Admin User</Text>
            </div>
          </Dropdown> */}
        </div>
      </Header>
      
      <Layout>
        {/* Sider */}
        <Sider breakpoint="lg" collapsedWidth="0" theme="dark" className="main-sider" style={{ position: 'relative', overflow: 'hidden' }}>
          <Menu
            mode="inline"
            theme="dark"
            selectedKeys={[selectedKey]}
            className="main-menu"
            items={menuItems}
            style={{ position: 'relative', zIndex: 1, background: 'transparent' }}
          />
        </Sider>
        {/* Main Content Area */}
        <Layout className="content-layout">
          <Content className="main-content" style={{ background: token.colorBgContainer, boxShadow: token.boxShadowTertiary, position: 'relative', overflow: 'hidden' }}>
            <div className="content-bg" aria-hidden="true">
              <svg className="content-rings" viewBox="0 0 1000 1000">
                <circle className="ring-pulse" cx="500" cy="500" r="130" />
                <circle cx="500" cy="500" r="210" />
                <circle className="ring-dashed" cx="500" cy="500" r="290" />
                <circle cx="500" cy="500" r="370" />
                <circle cx="500" cy="500" r="450" />
              </svg>
            </div>
            <BrandGlyph color="#E4007C" size={480} style={{ position: 'absolute', top: '68%', left: '78%', transform: 'translate(-50%, -50%) rotate(35deg)', opacity: 0.05, zIndex: 0 }} />
            <div style={{ position: 'relative', zIndex: 1 }}>
              <Outlet />
            </div>
          </Content>
          <Footer className="main-footer">
            HIDDEN ©{new Date().getFullYear()}
          </Footer>
        </Layout>
      </Layout>
    </Layout>
  );
};

export default MainLayout;
