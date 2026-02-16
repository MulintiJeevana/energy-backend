import nodemailer from "nodemailer";

export const sendOTPEmail = async (email, otp, name) => {
  try {
    const transporter = nodemailer.createTransport({
      host: "smtp-relay.brevo.com",
      port: 587,
      secure: false,
      auth: {
        user: process.env.BREVO_LOGIN,   // your brevo login email
        pass: process.env.BREVO_SMTP_KEY // smtp key from brevo
      }
    });

    await transporter.sendMail({
      from: `"Energy Monitor" <${process.env.BREVO_LOGIN}>`,
      to: email,
      subject: "OTP Verification",
      html: `
        <h2>Hello ${name}</h2>
        <p>Your OTP is:</p>
        <h1>${otp}</h1>
        <p>Valid for 5 minutes.</p>
      `
    });

  } catch (error) {
    console.error("Brevo SMTP error:", error.message);
    throw error;
  }
};





// import nodemailer from "nodemailer";

// export const sendOTPEmail = async (email, otp, name) => {
//   try {
//     const transporter = nodemailer.createTransport({
//       host: "smtp.gmail.com",
//       port: 587,
//       secure: false, // MUST be false for port 587
//       auth: {
//         user: process.env.EMAIL_USER,
//         pass: process.env.EMAIL_PASS, // App Password (not normal password)
//       },
//       tls: {
//         rejectUnauthorized: false,
//       },
//     });

//     const mailOptions = {
//       from: `"Energy Monitor" <${process.env.EMAIL_USER}>`,
//       to: email,
//       subject: "Energy Monitor - Email Verification",
//       html: `
//         <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
//           <h2 style="color: #3B82F6;">Welcome to Energy Monitor, ${name}!</h2>
//           <p>Please use the following OTP to verify your email address:</p>
//           <div style="background-color: #f3f4f6; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0;">
//             <h1 style="color: #1F2937; font-size: 32px; margin: 0; letter-spacing: 4px;">${otp}</h1>
//           </div>
//           <p>This OTP will expire in 5 minutes.</p>
//         </div>
//       `,
//     };

//     await transporter.sendMail(mailOptions);

//   } catch (error) {
//     console.error("Email send error:", error.message);
//     throw error;
//   }
// };

// import { Resend } from "resend";
// // re_J33Kvwsi_NQrpAFG3TsrrGa4ah17xKWfA



// export const sendOTPEmail = async (email, otp, name) => {
//   console.log(process.env.PORT)
//   const resend = new Resend(process.env.RESEND_API_KEY);

//   try {
//     console.log(email)
//     console.log(otp)
//     await resend.emails.send({
//       from: "onboarding@resend.dev", // works immediately
//       to: email,
//       subject: "Energy Monitor - Email Verification",
//       html: `
//         <div style="font-family: Arial;">
//           <h2>Welcome ${name}</h2>
//           <p>Your OTP is:</p>
//           <h1>${otp}</h1>
//           <p>This OTP expires in 5 minutes.</p>
//         </div>
//       `,
//     });
//   } catch (error) {
//     console.error("Resend error:", error.message);
//     throw error;
//   }
// };
