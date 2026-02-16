const requestLogger = (req, res, next) => {
    const startTime = Date.now();
    res.on("finish", () => {
        const duration = Date.now() - startTime
        const log = `Date ${new Date().toLocaleString()} | Ip ${req.ip} | ${req.method} ${req.originalUrl} ${res.statusCode} | Duration ${duration}ms\n`
        console.log(log)
    })
    next()
}
export default requestLogger;