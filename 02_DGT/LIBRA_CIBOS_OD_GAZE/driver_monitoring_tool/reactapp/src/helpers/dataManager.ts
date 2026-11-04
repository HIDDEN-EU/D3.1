import { getConfig } from './config';
import { keycloakInstance } from './keycloak';
import { notification } from 'antd';
import i18n from '../i18n';

export interface RequestOptions extends RequestInit {
  skipGlobalError?: boolean;
}

class DataManager {
  private get baseUrl() {
    return getConfig().apiBaseUrl;
  }

  private handleError(response: Response, skipGlobalError?: boolean) {
    if (skipGlobalError) return;

    let message = i18n.t('common.unexpected_error');
    let description = '';

    switch (response.status) {
      case 400:
        message = i18n.t('errors.bad_request');
        break;
      case 401:
        message = i18n.t('errors.unauthorized');
        // Optional: Trigger logout or redirect to login
        setTimeout(() => this.login(), 2000);
        break;
      case 403:
        message = i18n.t('errors.forbidden');
        break;
      case 404:
        message = i18n.t('errors.not_found');
        break;
      case 500:
        message = i18n.t('errors.server_error');
        break;
      default:
        description = `HTTP Error ${response.status}: ${response.statusText}`;
    }

    notification.error({
      message: i18n.t('errors.api_error_title'),
      description: description || message,
      placement: 'topRight',
      duration: 5,
    });
  }

  private async request(endpoint: string, options: RequestOptions = {}) {
    const config = getConfig();
    const url = `${this.baseUrl}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string>),
    };

    // Inject Keycloak token if enabled
    if (config.useKeycloak && keycloakInstance) {
      if (!keycloakInstance.token) {
        this.login();
        return new Promise(() => {}); // Halt execution
      }
      try {
        await keycloakInstance.updateToken(30);
        headers['Authorization'] = `Bearer ${keycloakInstance.token}`;
      } catch (err) {
        this.login();
        return new Promise(() => {}); // Halt execution
      }
    }

    try {
      const response = await fetch(url, { ...options, headers });

      if (!response.ok) {
        this.handleError(response, options.skipGlobalError);
        throw new Error(`HTTP ${response.status}`);
      }

      if (response.status === 204) return {};
      
      return response.json();
    } catch (error: any) {
      // Handle network errors (when fetch themselves fails)
      if (error.message === 'Failed to fetch' && !options.skipGlobalError) {
        notification.error({
          message: i18n.t('errors.api_error_title'),
          description: i18n.t('errors.network_error'),
        });
      }
      throw error;
    }
  }

  public get(endpoint: string, options?: RequestOptions) {
    return this.request(endpoint, { ...options, method: 'GET' });
  }

  public post(endpoint: string, data?: any, options?: RequestOptions) {
    return this.request(endpoint, { ...options, method: 'POST', body: JSON.stringify(data) });
  }

  public login = () => {
    if (getConfig().useKeycloak && keycloakInstance) keycloakInstance.login();
  };

  public logout = () => {
    if (getConfig().useKeycloak && keycloakInstance) keycloakInstance.logout();
  };
}

export const dataManager = new DataManager();
