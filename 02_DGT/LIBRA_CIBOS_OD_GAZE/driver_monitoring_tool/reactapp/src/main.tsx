import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import 'dayjs/locale/el';
import 'dayjs/locale/en';
import './index.css';
import App from './App.tsx';
import { setConfig } from './helpers/config.ts';
import { initKeycloak } from './helpers/keycloak.ts';
import './i18n';

// 1. Fetch the configuration first
fetch('/configuration/config.json')
  .then((response) => {
    if (!response.ok) {
      throw new Error(`Failed to load config.json: ${response.statusText}`);
    }
    return response.json();
  })
  .then(async (configData) => {
    // 2. Set the global config
    setConfig(configData);

    // 3. Initialize Keycloak if enabled before mounting
    if (configData.useKeycloak) {
      await initKeycloak();
    }

    // 4. Render the application safely now that config and auth are ready
    createRoot(document.getElementById('root')!).render(
      <StrictMode>
        <App />
      </StrictMode>,
    );
  })
  .catch((error) => {
    console.error('Initialization failed:', error);
    document.getElementById('root')!.innerHTML = `
      <div style="padding: 24px; color: red; font-family: sans-serif;">
        <h2>Initialization Error</h2>
        <p>Could not initialize the application.</p>
        <pre>${error.message}</pre>
      </div>
    `;
  });
