import twilio from "twilio";


export const sendSMS = async (phone, otp) => {
    const client = twilio(
        process.env.TWILIO_ACCOUNT_SID,
        process.env.TWILIO_AUTH_TOKEN
    );
    console.log("sending sms")
    // console.log(process.env.TWILIO_MESSAGING_SERVICE_SID)
    return client.messages.create({
        body: `Your OTP is ${otp}`,
        messagingServiceSid: process.env.TWILIO_MESSAGING_SERVICE_SID,
        to: phone
    });
};
