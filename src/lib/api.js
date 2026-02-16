import axios from 'axios';

/**
 * Centralized API configuration.
 *
 * The base URL is read from the VITE_API_BASE_URL environment variable,
 * which is set in .env.development (localhost) and .env.production (render).
 *
 * Usage:
 *   import { api, apiFetch } from '@/lib/api';
 *
 *   // Axios — for AuthContext and similar:
 *   const res = await api.get('/api/auth/profile');
 *
 *   // Fetch wrapper — drop-in replacement for fetch() calls:
 *   const res = await apiFetch('/api/usage/hourly?date=2024-03-03');
 *   const data = await res.json();
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001';

/** Pre-configured Axios instance with baseURL and credentials. */
export const api = axios.create({
    baseURL: BASE_URL,
    withCredentials: true,
});

/**
 * Thin wrapper around fetch() that prepends the base URL
 * and sets credentials: 'include' by default.
 */
export const apiFetch = (path, options = {}) => {
    return fetch(`${BASE_URL}${path}`, {
        credentials: 'include',
        ...options,
    });
};

export default api;
