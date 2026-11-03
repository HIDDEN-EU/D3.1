import Keycloak from 'keycloak-js';

export let keycloakInstance: Keycloak | null = null;

export const initKeycloak = async (configPath: string = '/configuration/keycloak.json'): Promise<void> => {
  try {
    const response = await fetch(configPath);
    if (!response.ok) {
      throw new Error(`Failed to load keycloak configuration from ${configPath}`);
    }
    const kcConfig = await response.json();
    
    keycloakInstance = new Keycloak(kcConfig);
    
    // Initialize Keycloak
    const authenticated = await keycloakInstance.init({
      onLoad: 'check-sso', // Validate SSO session silently without forcing redirect unless needed
      checkLoginIframe: false, // Often disabled to prevent modern browser iframe blocking issues in templates
    });
    
    console.log('Keycloak Initialized. Authenticated:', authenticated);
  } catch (error) {
    console.error('Failed to initialize Keycloak:', error);
  }
};
