import React from 'react';
import { Toaster } from './components/ui/toaster';
import { Toaster as Sonner } from './components/ui/sonner';
import { TooltipProvider } from './components/ui/tooltip';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

import { ThemeProvider } from './context/ThemeContext';
import { AuthProvider, useAuth } from './context/AuthContext';
import Navbar from './components/Navbar';

import Home from './pages/Home';
import Appliances from './pages/Appliances';
import Recommendations from './pages/Recommendations';
import About from './pages/About';
import Settings from './pages/Settings';
import NotFound from './pages/NotFound';
import Login from './pages/Login';
import Signup from './pages/Signup';
import OTPVerification from './pages/OTPVerification'
import Analytics from './pages/Analytics';

const queryClient = new QueryClient();

/** ✅ Protected Route Logic */
const ProtectedRoute = ({ element }) => {
  const { user, loading } = useAuth();

  if (loading) return <div>Loading...</div>;

  return user ? element : <Navigate to="/login" replace />;
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <AuthProvider>
      <ThemeProvider>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <div className="min-h-screen">
              <Navbar />
              <Routes>
                {/* ✅ Public Routes */}
                <Route path="/about" element={<About />} />
                <Route path="/login" element={<Login />} />
                <Route path="/signup" element={<Signup />} />
                <Route path='/verify-otp' element={<OTPVerification />} />

                {/* ✅ Protected Routes */}
                <Route path="/" element={<ProtectedRoute element={<Home />} />} />
                <Route path="/appliances" element={<ProtectedRoute element={<Appliances />} />} />
                <Route path="/recommendations" element={<ProtectedRoute element={<Recommendations />} />} />
                <Route path="/settings" element={<ProtectedRoute element={<Settings />} />} />
                <Route path="/analytics" element={<ProtectedRoute element={<Analytics />} />} />

                {/* ✅ Catch-all */}
                <Route path="*" element={<NotFound />} />
              </Routes>
            </div>
          </BrowserRouter>
        </TooltipProvider>
      </ThemeProvider>
    </AuthProvider>
  </QueryClientProvider>
);

export default App;