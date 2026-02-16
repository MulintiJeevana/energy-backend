import express from 'express';
import cors from 'cors';

import dotenv from 'dotenv';
dotenv.config();
import session from 'express-session';
import MongoStore from 'connect-mongo';
import mongoose from 'mongoose';
import authRoutes from './routes/auth.js';
import deviceRoutes from './routes/devices.js';
import energyRoutes from './routes/energy.js';
import usageRoutes from "./routes/usage.js";
import analyticsRoutes from "./routes/analytics.js";
import requestLogger from './middleware/logger.js'



const app = express();
const PORT = process.env.PORT || 3001;

// Trust first proxy (Render, Heroku, etc.) so secure cookies work behind TLS termination
app.set('trust proxy', 1);


// Database connection
console.log(process.env.MONGO_URI)
mongoose.connect(process.env.MONGO_URI)
  .then(() => {
    console.log("connected to the database ");
  })
  .catch((err) => {
    console.log("connection err : ", err)
  })


// Middleware
app.use(cors({
  origin: [
    'http://localhost:5173',
    'https://energy-tracking-app.onrender.com',
    'https://energy-tracking-app-mauve.vercel.app'


  ],
  credentials: true
}));

app.use(express.json());

const isProduction = process.env.NODE_ENV === 'production' || !!process.env.RENDER;
console.log('Environment:', { NODE_ENV: process.env.NODE_ENV, RENDER: process.env.RENDER, isProduction });

// Session configuration
app.use(session({
  secret: process.env.SESSION_SECRET,
  resave: false,
  saveUninitialized: false,
  store: MongoStore.create({
    mongoUrl: process.env.MONGO_URI,
    touchAfter: 24 * 3600 // lazy session update
  }),
  cookie: {
    secure: isProduction,
    httpOnly: true,
    sameSite: isProduction ? 'none' : 'lax',
    maxAge: 1000 * 60 * 60 * 24 * 7 // 7 days
  }
}));

app.use(requestLogger);

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/devices', deviceRoutes);
app.use('/api/energy', energyRoutes);
app.use("/api/usage", usageRoutes);
app.use("/api/analytics", analyticsRoutes);


app.get('/api/health', (req, res) => {
  res.json({ status: 'Server is running!' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
