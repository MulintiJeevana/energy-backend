import React from 'react';
import { NavLink, Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useTheme } from '../context/ThemeContext';
import { useAuth } from '../context/AuthContext';
import { Button } from '../components/ui/button';
import { Home, Settings, Sun, Moon, LogOut, User } from 'lucide-react';

const Navbar = () => {
  const { theme, toggleTheme } = useTheme();
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      await logout(); // clear session
      navigate('/'); // back to public page
    } catch (err) {
      console.error('Logout failed:', err);
    }
  };

  const navItems = [
    ...(user ? [
      { path: '/', label: 'Home', icon: Home },
      { path: '/appliances', label: 'Appliances' },
      { path: '/recommendations', label: 'Recommendations' },
      { path: '/analytics', label: 'Analytics' },
      { path: '/settings', label: 'Settings', icon: Settings },
    ] : []),
    { path: '/about', label: 'About' }
  ];

  return (
    <nav className="bg-card/80 backdrop-blur-lg border-b border-border sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">

          {/* Logo */}
          <motion.div
            className="flex items-center space-x-3"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="w-8 h-8 energy-gradient rounded-lg flex items-center justify-center">
              <div className="w-4 h-4 bg-white rounded-sm"></div>
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-energy-primary to-energy-secondary bg-clip-text text-transparent">
              EnergyMonitor
            </span>
          </motion.div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-6">
            {navItems.map((item, index) => (
              <motion.div
                key={item.path}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <NavLink
                  to={item.path}
                  className={({ isActive }) =>
                    `relative px-3 py-2 text-sm font-medium transition-colors duration-200 flex items-center space-x-2 ${isActive
                      ? 'text-energy-primary'
                      : 'text-muted-foreground hover:text-foreground'
                    }`
                  }
                >
                  {({ isActive }) => (
                    <>
                      {item.icon && <item.icon className="w-4 h-4" />}
                      <span>{item.label}</span>
                      {isActive && (
                        <motion.div
                          className="absolute -bottom-2 left-0 right-0 h-0.5 bg-energy-primary rounded-full"
                          layoutId="activeTab"
                          initial={false}
                          transition={{ type: "spring", stiffness: 300, damping: 30 }}
                        />
                      )}
                    </>
                  )}
                </NavLink>
              </motion.div>
            ))}

            {/* Auth buttons */}
            {!user ? (
              <>
                <Link to="/login" className="text-sm text-energy-primary hover:underline">Login</Link>
                <Link to="/signup" className="text-sm text-white bg-energy-primary px-3 py-1 rounded-md hover:opacity-90">
                  Sign Up
                </Link>
              </>
            ) : (
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-1 text-sm text-muted-foreground">
                  <User className="w-4 h-4" />
                  <span>{user.name}</span>
                </div>
                <Button onClick={handleLogout} className="text-red-500 hover:text-red-600 text-sm" variant="ghost">
                  <LogOut className="w-4 h-4 mr-1" />
                  Logout
                </Button>
              </div>
            )}
          </div>

          {/* Theme Toggle */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
              className="hover:bg-accent"
            >
              {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </Button>
          </motion.div>
        </div>
      </div>

      {/* Mobile Navigation */}
      <div className="md:hidden border-t border-border px-4 pb-4 pt-2 space-y-2">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `block px-3 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${isActive
                ? 'bg-accent text-energy-primary'
                : 'text-muted-foreground hover:text-foreground hover:bg-accent/50'
              }`
            }
          >
            <div className="flex items-center space-x-2">
              {item.icon && <item.icon className="w-4 h-4" />}
              <span>{item.label}</span>
            </div>
          </NavLink>
        ))}

        {!user ? (
          <div className="flex flex-col space-y-2">
            <Link to="/login" className="text-sm text-energy-primary hover:underline">Login</Link>
            <Link to="/signup" className="text-sm text-white bg-energy-primary px-3 py-1 rounded-md hover:opacity-90 text-center">
              Sign Up
            </Link>
          </div>
        ) : (
          <div className="flex items-center justify-between">
            <span className="text-sm flex items-center text-muted-foreground space-x-1">
              <User className="w-4 h-4" />
              <span>{user.name}</span>
            </span>
            <Button onClick={handleLogout} className="text-red-500 text-sm" variant="ghost">
              <LogOut className="w-4 h-4 mr-1" />
              Logout
            </Button>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;



