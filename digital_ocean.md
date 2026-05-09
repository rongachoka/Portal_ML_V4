# Normal connection
ssh root@68.183.0.188

# When port 22 is blocked (office networks, restricted wifi)
ssh -p 443 root@68.183.0.188

# To look at connections / ip addresses
sudo nano /etc/postgresql/17/main/pg_hba.conf


# -------------
# -------------

# When you need to connect from your Windows machine/ Macbook, you'd SSH tunnel like this:
ssh -L 5432:localhost:5432 root@68.183.0.188

Right now your PostgreSQL only accepts connections from the droplet itself — no outside connections allowed. But sometimes you need to connect from your laptop (Windows or Mac) to view data, run queries, or use DBeaver/pgAdmin.
The SSH tunnel solves this by creating a secure pipe:
Your laptop (port 5432) ──SSH tunnel──► Droplet ──► PostgreSQL (port 5432)
When you run:
bashssh -L 5432:localhost:5432 root@68.183.0.188
It means:

5432 (first one) — listen on port 5432 on your laptop
localhost:5432 — forward to localhost:5432 on the droplet
root@68.183.0.188 — through this SSH connection

So when DBeaver on your laptop connects to localhost:5432, it's actually talking to PostgreSQL on the droplet through the encrypted SSH tunnel. From PostgreSQL's perspective the connection looks like it's coming from 127.0.0.1 — which is allowed.


##

When it restarts:

If the droplet reboots (e.g. DigitalOcean maintenance, power issue)
If cloudflared crashes and systemd restarts it
If you manually restart it

In practice on a stable droplet this might happen once a month or less. But when it does, the URL changes and you'd need to:

Run journalctl -u cloudflared-quick | grep "trycloudflare.com" to get the new URL
Update the webhook URL in Respond.io settings
Update the WEBHOOK_URL in the n8n Docker container

That's annoying but manageable for a short period.