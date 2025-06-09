# Banking ML Platform API Usage Guide

This document provides comprehensive examples for using the Banking ML Platform API endpoints.

## Base URL

- Local Development: `http://localhost:8000`
- Production: `https://your-production-url.com`

## Authentication

The API uses OAuth2 password flow for authentication.

### Get Access Token

```bash
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=analyst&password=analyst123"
