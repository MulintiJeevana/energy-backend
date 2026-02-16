import { MongoClient } from "mongodb";
import dotenv from "dotenv";
dotenv.config();

const uri = process.env.MONGO_URI;
let client;

async function getClient() {
    if (!client) {
        if (!uri) throw new Error("MONGO_URI is not defined. Check your .env file.");
        client = new MongoClient(uri);
        await client.connect();
    }
    return client;
}

/**
 * GET /api/analytics/daily-usage?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD
 * Returns daily energy totals for the given date range.
 */
export const getDailyUsage = async (req, res) => {
    try {
        const { startDate, endDate } = req.query;

        if (!startDate || !endDate) {
            return res.status(400).json({ error: "startDate and endDate are required (YYYY-MM-DD)" });
        }

        const start = new Date(startDate + "T00:00:00.000Z");
        const end = new Date(endDate + "T23:59:59.999Z");

        if (isNaN(start.getTime()) || isNaN(end.getTime())) {
            return res.status(400).json({ error: "Invalid date format. Use YYYY-MM-DD." });
        }

        const dbClient = await getClient();
        const collection = dbClient.db("energyDB").collection("energy");

        const pipeline = [
            { $match: { date: { $gte: start, $lte: end } } },
            {
                $group: {
                    _id: {
                        day: { $dayOfMonth: "$date" },
                        month: { $month: "$date" },
                        year: { $year: "$date" },
                    },
                    usage: { $sum: "$energy_consumed" },
                },
            },
            { $sort: { "_id.year": 1, "_id.month": 1, "_id.day": 1 } },
            {
                $project: {
                    _id: 0,
                    date: {
                        $dateToString: {
                            format: "%Y-%m-%d",
                            date: {
                                $dateFromParts: {
                                    year: "$_id.year",
                                    month: "$_id.month",
                                    day: "$_id.day",
                                },
                            },
                        },
                    },
                    usage: { $round: ["$usage", 2] },
                },
            },
        ];

        const data = await collection.aggregate(pipeline).toArray();
        res.json({ success: true, data });
    } catch (err) {
        console.error("Error in /analytics/daily-usage:", err.message);
        res.status(500).json({ error: "Failed to fetch daily usage analytics" });
    }
};

/**
 * GET /api/analytics/device-usage?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD
 * Returns total energy consumption grouped by device for the given date range.
 */
export const getDeviceUsage = async (req, res) => {
    try {
        const { startDate, endDate } = req.query;

        if (!startDate || !endDate) {
            return res.status(400).json({ error: "startDate and endDate are required (YYYY-MM-DD)" });
        }

        const start = new Date(startDate + "T00:00:00.000Z");
        const end = new Date(endDate + "T23:59:59.999Z");

        if (isNaN(start.getTime()) || isNaN(end.getTime())) {
            return res.status(400).json({ error: "Invalid date format. Use YYYY-MM-DD." });
        }

        const dbClient = await getClient();
        const collection = dbClient.db("energyDB").collection("energy");

        const pipeline = [
            { $match: { date: { $gte: start, $lte: end } } },
            {
                $group: {
                    _id: "$device_id",
                    usage: { $sum: "$energy_consumed" },
                },
            },
            { $sort: { usage: -1 } },
            {
                $project: {
                    _id: 0,
                    device: "$_id",
                    usage: { $round: ["$usage", 2] },
                },
            },
        ];

        const data = await collection.aggregate(pipeline).toArray();
        res.json({ success: true, data });
    } catch (err) {
        console.error("Error in /analytics/device-usage:", err.message);
        res.status(500).json({ error: "Failed to fetch device usage analytics" });
    }
};
