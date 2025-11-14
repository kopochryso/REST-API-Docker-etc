import express from 'express';

const app = express();

app.get('/', (req, res) => {
    res.send(`WELCOME TO A TERRIBLE DOCKER TUTORIAL`);
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
    console.log(`Unfortunately listening on port http://localhost:${port}`);
});
