import React, { createContext, useContext, useState, useEffect } from 'react';
import { api } from '../lib/api';

const AuthContext = createContext();
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = async () => {
    try {
      const response = await api.get('/api/auth/profile');
      if (response.data.success) {
        setUser(response.data.user);
      }
    } catch (error) {
      console.error('Auth check failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const login = async (email, password) => {
    try {
      const response = await api.post('/api/auth/login', { email, password });
      setUser(response.data.user);
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Login failed');
    }
  };

  const logout = async () => {
    try {
      await api.post('/api/auth/logout');
      setUser(null);
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  const signup = async (email, password, name, phone) => {
    try {
      await api.post('/api/auth/signup', { email, password, name, phone });
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Signup failed');
    }
  };

  const verifyOTP = async (email, otp) => {
    try {
      await api.post('/api/auth/verify-otp', { email, otp });
    } catch (error) {
      throw new Error(error.response?.data?.message || 'OTP verification failed');
    }
  };

  const resendOTP = async (email) => {
    try {
      await api.post('/api/auth/resend-otp', { email });
    } catch (error) {
      throw new Error(error.response?.data?.message || 'OTP resend failed');
    }
  };

  const value = {
    user,
    login,
    logout,
    signup,
    verifyOTP,
    resendOTP,
    loading,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
