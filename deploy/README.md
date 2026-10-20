# Reference proxy services

Optional Nginx and HAProxy services over the same four backends, used as
mature-proxy baselines next to the Python balancer. They are deliberately not
in `docker-compose.yml`: the stack under test should not quietly gain two more
competing proxies.

```bash
docker compose -f docker-compose.yml -f deploy/compose.reference.yml up -d
curl http://localhost:8081/ping   # Nginx
curl http://localhost:8082/ping   # HAProxy
```

Inside the compose network they listen on `http://nginx:8080` and
`http://haproxy:8080`, which is the address the load generator would target
(`bench/quick_dataset.py` targets `http://balancer:8080`).

| | Python balancer | Nginx | HAProxy |
|---|---|---|---|
| Method | predicted latency x in-flight | `least_conn` | `leastconn` |
| Health | active metrics collector + circuit breaker | passive `max_fails` | active `httpchk` every 2 s |
| Retry on 5xx | one retry on another server | no (by default) | no (connection failures only) |
| Slow start | yes, 10 s ramp | no | no |
| Metrics / request log | Parquet per run + `/stats` | access log | access log |

## Honest status

These configs are provided as Phase 6 deliverables and are validated by
starting them and issuing requests through them. They are **not** part of the
Phase 7 campaign: every number in `docs/results.md` is measured from the
balancer's own request log, and Nginx/HAProxy write no such log, so a
like-for-like comparison needs a client-side measurement harness. That harness
is listed as an open item in the README rather than approximated here.
