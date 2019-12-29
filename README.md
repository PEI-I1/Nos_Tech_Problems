# Nos_Tech_Problems

## Table of Contents
* [What is it?](#what-is-it)
  - [Features](#features)
* [Usage](#usage)
  - [Development setup](#development-setup)
  - [Deployment](#deployment)
* [Architecture](#architecture)
  - [Overview](#overview)
  - [API](#api)


## What is it?
NOS Tech Problems is a bot module that combines Natural Language Processing capabilities 
with a machine learning backend to provide technical assistance to users. By taking advantage
of past user interactions, whether with real/human assistants or with the bot module itself, it
improves the model's ability to provide accurate solutions to users problems.

### Features


## Usage
### Development Setup

### Deployment


## Architecture
### Overview
The microservice is made up of two different sub-modules: 
- the NTP Bot, which is responsible for handling incoming client messages and responding
- the Tech Problems sub-module responsible for authenticating clients and proposing solutions to their problems

hence only the NTP Bot is publically exposed, handling all the interaction between the users and the underlying
system:

![NOS\_Tech\_Problems Overview](static/NTP_Overview.png)

### API
The only public API endpoint provided by the NOS Tech Problems microservice that is 
exposed by the [NTP Bot](NTP_Bot) however, the following list includes the endpoints made
available by the [Tech Problems module](Tech_Problems) to facilitate future development.

#### NTP Bot
<details>
<summary>Public API endpoint that allows user interaction with the NTP Bot</summary>
```http
POST /solver
```
</details>

#### NOS Tech Problems
In order to authenticate users this module uses Django's authentication backend and Session management features, 
which requires all requests made to the API to take advantage of 
[HTTP sessions](https://developer.mozilla.org/en-US/docs/Web/HTTP/Session).

<details>
<summary></summary>
```http
GET /problems/login?username=<>&password=<>
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `username` | `string` | Username (telephone number) |
| `password` | `string` | Password (NIF) |

</details>

<details>
<summary>Clears a users session data</summary>
```http
GET /problems/logout
```
</details>

<details>
<summary>Provides a solution for a problem described by the input parameters</summary>
```http
GET /problems/solve?sintoma=<>&tipificacao_tipo_1=<>&tipificacao_tipo_2=<>&tipificacao_tipo_3=<>
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `sintoma` | `string` | Internal description of the problem's simptome |
| `tipificacao_tipo_1` | `string` | Problem Typification - Type 1 |
| `tipificacao_tipo_2` | `string` | Problem Typification - Type 2 |
| `tipificacao_tipo_3` | `string` | Problem Typification - Type 3 |

</details>

<details>
<summary>Register a new user/client in the NOS Tech Problems backend</summary>
```http
GET /problems/register?username=<>&password=<>&morada=<>&equipamentos=<>&tarifario=<>
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `username` | `string` | Username (telephone number) |
| `password` | `string` | Password (NIF) |
| `morada` | `string` | Client's address |
| `equipamentos` | `string` | Client's devices that pertain to the ISP service |
| `tarifario` | `string` | Client's contracted tariff |
</details>
