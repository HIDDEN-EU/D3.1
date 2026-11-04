export interface AppConfig {
  apiBaseUrl: string;
  janusUrl: string;
  rosbridgeUrl: string;
  appName: string;
  themeColor: string;
  version: string;
  useKeycloak: boolean;
}

// Global variable to hold the config
let config: AppConfig | null = null;

// Sets the config (called once during initialization)
export const setConfig = (newConfig: AppConfig) => {
  config = newConfig;
};

// Gets the config anywhere in the app
export const getConfig = (): AppConfig => {
  if (!config) {
    throw new Error("Configuration has not been loaded yet!");
  }
  return config;
};
