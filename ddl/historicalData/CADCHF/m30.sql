CREATE TABLE "historicalData".cadchf_m30 (
                           date        DATE        NOT NULL,
                           time        TIME        NOT NULL,
                           open        NUMERIC(10, 5) NOT NULL,
                           high        NUMERIC(10, 5) NOT NULL,
                           low         NUMERIC(10, 5) NOT NULL,
                           close       NUMERIC(10, 5) NOT NULL,
                           tick_volume INTEGER     NULL,
                           PRIMARY KEY (date, time)
);

CREATE INDEX idx_cadchf_m30_date ON "historicalData".cadchf_m30 (date);
COMMENT ON TABLE "historicalData".cadchf_m30 IS '30-minute OHLC bar data. Timezone: GMT+2 with US DST applied. Source: Dukascopy export from Tick Data Suite.';
/* date,time,open,high,low,close,tick_volume */