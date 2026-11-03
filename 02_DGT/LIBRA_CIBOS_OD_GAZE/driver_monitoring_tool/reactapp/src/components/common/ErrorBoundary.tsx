import { Component, type ErrorInfo, type ReactNode } from 'react';
import { Result, Button, Typography, Space, Collapse } from 'antd';
import { ReloadOutlined, HomeOutlined, BugOutlined } from '@ant-design/icons';
import i18n from '../../i18n';

const { Text } = Typography;
const { Panel } = Collapse;

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI.
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // You can also log the error to an error reporting service
    console.error('Uncaught error:', error, errorInfo);
    this.setState({ errorInfo });
  }

  private handleReload = () => {
    window.location.reload();
  };

  private handleGoHome = () => {
    window.location.href = '/';
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          minHeight: '100vh',
          padding: '24px',
          background: 'inherit' 
        }}>
          <Result
            status="error"
            title={i18n.t('error_boundary.title')}
            subTitle={i18n.t('error_boundary.subtitle')}
            extra={[
              <Button 
                type="primary" 
                key="console" 
                icon={<ReloadOutlined />} 
                onClick={this.handleReload}
              >
                {i18n.t('error_boundary.reload_button')}
              </Button>,
              <Button 
                key="buy" 
                icon={<HomeOutlined />} 
                onClick={this.handleGoHome}
              >
                {i18n.t('error_boundary.home_button')}
              </Button>,
            ]}
          >
            <div className="desc">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text strong style={{ fontSize: 16 }}>
                  <BugOutlined style={{ marginRight: 8, color: '#ff4d4f' }} />
                  {i18n.t('error_boundary.technical_details')}
                </Text>
                
                <Collapse ghost>
                  <Panel header={this.state.error?.toString() || 'Unknown Error'} key="1">
                    <pre style={{ 
                      whiteSpace: 'pre-wrap', 
                      wordBreak: 'break-all', 
                      fontSize: '12px',
                      maxHeight: '300px',
                      overflow: 'auto',
                      background: 'rgba(0,0,0,0.02)',
                      padding: '12px',
                      borderRadius: '8px'
                    }}>
                      {this.state.errorInfo?.componentStack}
                    </pre>
                  </Panel>
                </Collapse>
              </Space>
            </div>
          </Result>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
