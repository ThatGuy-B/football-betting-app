import streamlit as st
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import datetime

# API Key from secrets
API_KEY = st.secrets['API_KEY']

# Fetch function with improved error logging
def fetch_api(endpoint, params={}):
    url = f"https://v3.football.api-sports.io/{endpoint}"
    headers = {
        'x-rapidapi-host': 'v3.football.api-sports.io',
        'x-rapidapi-key': API_KEY
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json().get('response', [])
        if not data:
            st.warning(f"Empty response from API for endpoint: {endpoint}, params: {params}")
        return data
    else:
        st.error(f"API error for endpoint {endpoint}: {response.status_code} - {response.text}")
        return []

# League options
leagues = {
    39: "English Premier League",
    140: "La Liga",
    61: "Ligue 1",
    78: "Bundesliga",
    135: "Serie A"
}

# Streamlit UI
st.title("Football Betting Predictions")
league_id = st.selectbox("Choose a League", options=list(leagues.keys()), format_func=lambda x: leagues[x])
season = datetime.datetime.now().year

# Fetch and prepare data
@st.cache_data(ttl=3600)
def get_predictions(league_id, season):
    historical_matches = []
    for past_season in range(season - 3, season):
        matches = fetch_api('fixtures', {'league': league_id, 'season': past_season})
        historical_matches.extend(matches)

    data = []
    for match in historical_matches:
        if match['fixture']['status']['short'] != 'FT': 
            continue
        match_id = match['fixture']['id']
        home_id = match['teams']['home']['id']
        away_id = match['teams']['away']['id']
        date = pd.to_datetime(match['fixture']['date'])

        stats = fetch_api('fixtures/statistics', {'fixture': match_id}) or [{}]*2
        # Skip matches with missing or invalid statistics
        if not stats or not stats[0].get('statistics') or (len(stats) > 1 and not stats[1].get('statistics')):
            st.warning(f"Skipping match {match_id} due to missing statistics")
            continue

        lineups = fetch_api('fixtures/lineups', {'fixture': match_id}) or []
        injuries = fetch_api('injuries', {'league': league_id, 'season': past_season}) or []
        h2h = fetch_api('fixtures/headtohead', {'h2h': f"{home_id}-{away_id}"}) or []
        events = fetch_api('fixtures/events', {'fixture': match_id}) or []

        home_stats = stats[0].get('statistics', [])
        away_stats = stats[1].get('statistics', []) if len(stats) > 1 else []
        possession_home = next((s['value'] for s in home_stats if s['type'] == 'Ball Possession'), 50)
        shots_home = next((s['value'] for s in home_stats if s['type'] == 'Total Shots'), 0)
        corners_home = next((s['value'] for s in home_stats if s['type'] == 'Corner Kicks'), 0)
        cards_home = next((s['value'] for s in home_stats if s['type'] == 'Yellow Cards'), 0)
        fouls_home = next((s['value'] for s in home_stats if s['type'] == 'Fouls'), 0)
        shots_on_target_home = next((s['value'] for s in home_stats if s['type'] == 'Shots on Goal'), 0)
        offsides_home = next((s['value'] for s in home_stats if s['type'] == 'Offsides'), 0)
        home_injuries = sum(1 for i in injuries if i['player']['team'] == home_id)

        # Half-time stats (approximate from events if available)
        first_half_goals = sum(1 for e in events if e['type'] == 'Goal' and e['time']['elapsed'] <= 45)
        second_half_goals = sum(1 for e in events if e['type'] == 'Goal' and e['time']['elapsed'] > 45)

        row = {
            'Date': date,
            'HomeTeam': match['teams']['home']['name'],
            'AwayTeam': match['teams']['away']['name'],
            'HomeID': home_id,
            'AwayID': away_id,
            'FTHG': match['goals']['home'],
            'FTAG': match['goals']['away'],
            'FTR': 'H' if match['goals']['home'] > match['goals']['away'] else ('A' if match['goals']['home'] < match['goals']['away'] else 'D'),
            'possession_home': possession_home,
            'shots_home': shots_home,
            'corners_home': corners_home,
            'cards_home': cards_home,
            'fouls_home': fouls_home,
            'shots_on_target_home': shots_on_target_home,
            'offsides_home': offsides_home,
            'home_injuries': home_injuries,
            'first_half_goals': first_half_goals,
            'second_half_goals': second_half_goals,
            'goal_diff': match['goals']['home'] - match['goals']['away'],
            'clean_sheet_home': 1 if match['goals']['away'] == 0 else 0,
        }
        data.append(row)

    df = pd.DataFrame(data)
    le = LabelEncoder()
    df['HomeTeam'] = le.fit_transform(df['HomeTeam'])
    df['AwayTeam'] = le.transform(df['AwayTeam'])

    def calculate_form(team_id, date, matches, n=5):
        past = sorted([m for m in matches if pd.to_datetime(m['fixture']['date']) < date and (m['teams']['home']['id'] == team_id or m['teams']['away']['id'] == team_id)], key=lambda x: x['fixture']['date'])[-n:]
        points = 0
        for m in past:
            if m['teams']['home']['id'] == team_id:
                if m['goals']['home'] > m['goals']['away']: points += 3
                elif m['goals']['home'] == m['goals']['away']: points += 1
            else:
                if m['goals']['away'] > m['goals']['home']: points += 3
                elif m['goals']['away'] == m['goals']['home']: points += 1
        return points / max(1, len(past))

    def h2h_wins(home_id, away_id, date, h2h_matches, n=5):
        past_h2h = sorted([m for m in h2h_matches if pd.to_datetime(m['fixture']['date']) < date], key=lambda x: x['fixture']['date'])[-n:]
        home_wins = sum(1 for m in past_h2h if 
                        (m['teams']['home']['id'] == home_id and m['goals']['home'] > m['goals']['away']) or 
                        (m['teams']['away']['id'] == home_id and m['goals']['away'] > m['goals']['home']))
        return home_wins

    for i, row in df.iterrows():
        df.at[i, 'home_form'] = calculate_form(row['HomeID'], row['Date'], historical_matches)
        df.at[i, 'away_form'] = calculate_form(row['AwayID'], row['Date'], historical_matches)
        h2h_list = fetch_api('fixtures/headtohead', {'h2h': f"{row['HomeID']}-{row['AwayID']}", 'last': 5})
        df.at[i, 'h2h_home_wins'] = h2h_wins(row['HomeID'], row['AwayID'], row['Date'], h2h_list)

    df = df.dropna()
    features = ['HomeTeam', 'AwayTeam', 'home_form', 'away_form', 'h2h_home_wins', 'possession_home', 'shots_home', 'home_injuries']

    # Train models for each market
    models = {}
    X = df[features]

    # Match Winner
    y_winner = le.fit_transform(df['FTR'])
    models['winner'] = RandomForestClassifier(n_estimators=100, random_state=42)
    models['winner'].fit(X, y_winner)

    # Total Goals
    y_goals = df['FTHG'] + df['FTAG']
    models['goals'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models['goals'].fit(X, y_goals)

    # Corners
    y_corners = df['corners_home'] + [
        next((s['value'] for s in stats[1].get('statistics', []) if s['type'] == 'Corner Kicks'), 0)
        for stats in [
            fetch_api('fixtures/statistics', {'fixture': m['fixture']['id']}) or [{}]*2
            for m in historical_matches[:len(df)]
        ]
    ]
    models['corners'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models['corners'].fit(X, y_corners[:len(X)])

    # Cards
    y_cards = df['cards_home'] + [
        next((s['value'] for s in stats[1].get('statistics', []) if s['type'] == 'Yellow Cards'), 0)
        for stats in [
            fetch_api('fixtures/statistics', {'fixture': m['fixture']['id']}) or [{}]*2
            for m in historical_matches[:len(df)]
        ]
    ]
    models['cards'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models['cards'].fit(X, y_cards[:len(X)])

    # First Half Goals
    models['first_half_goals'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models['first_half_goals'].fit(X, df['first_half_goals'])

    # Second Half Goals
    models['second_half_goals'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models['second_half_goals'].fit(X, df['second_half_goals'])

    # Asian Handicap (predict goal difference)
    models['goal_diff'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models['goal_diff'].fit(X, df['goal_diff'])

    # Clean Sheet (Home)
    models['clean_sheet_home'] = RandomForestClassifier(n_estimators=100, random_state=42)
    models['clean_sheet_home'].fit(X, df['clean_sheet_home'])

    # Fetch upcoming matches
    upcoming = fetch_api('fixtures', {'league': league_id, 'season': season, 'next': 10})
    predictions = []
    for match in upcoming:
        home_team = match['teams']['home']['name']
        away_team = match['teams']['away']['name']
        home_id = match['teams']['home']['id']
        away_id = match['teams']['away']['id']
        date = pd.to_datetime(match['fixture']['date'])

        injuries = fetch_api('injuries', {'league': league_id, 'season': season}) or []
        h2h = fetch_api('fixtures/headtohead', {'h2h': f"{home_id}-{away_id}", 'last': 5})

        home_form = calculate_form(home_id, date, historical_matches + upcoming)
        away_form = calculate_form(away_id, date, historical_matches + upcoming)
        h2h_wins_val = h2h_wins(home_id, away_id, date, h2h)
        possession_home = 50
        shots_home = 10
        home_injuries_val = sum(1 for i in injuries if i['player']['team'] == home_id)

        new_match = pd.DataFrame({
            'HomeTeam': le.transform([home_team])[0] if home_team in le.classes_ else 0,
            'AwayTeam': le.transform([away_team])[0] if away_team in le.classes_ else 0,
            'home_form': home_form,
            'away_form': away_form,
            'h2h_home_wins': h2h_wins_val,
            'possession_home': possession_home,
            'shots_home': shots_home,
            'home_injuries': home_injuries_val,
        }, index=[0])

        # Predictions for all markets
        winner_pred = models['winner'].predict(new_match)[0]
        goals_pred = models['goals'].predict(new_match)[0]
        corners_pred = models['corners'].predict(new_match)[0]
        cards_pred = models['cards'].predict(new_match)[0]
        first_half_goals_pred = models['first_half_goals'].predict(new_match)[0]
        second_half_goals_pred = models['second_half_goals'].predict(new_match)[0]
        goal_diff_pred = models['goal_diff'].predict(new_match)[0]
        clean_sheet_pred = models['clean_sheet_home'].predict(new_match)[0]

        # Double chance (derived from winner)
        winner_prob = models['winner'].predict_proba(new_match)[0]
        double_chance = {
            '1X': winner_prob[le.transform(['H'])[0]] + winner_prob[le.transform(['D'])[0]] if 'H' in le.classes_ and 'D' in le.classes_ else 0,
            'X2': winner_prob[le.transform(['D'])[0]] + winner_prob[le.transform(['A'])[0]] if 'D' in le.classes_ and 'A' in le.classes_ else 0,
            '12': winner_prob[le.transform(['H'])[0]] + winner_prob[le.transform(['A'])[0]] if 'H' in le.classes_ and 'A' in le.classes_ else 0
        }
        double_chance_pred = max(double_chance, key=double_chance.get)

        # First scorer (simplified: pick player with most goals in squad)
        lineups = fetch_api('fixtures/lineups', {'fixture': match['fixture']['id']}) or []
        top_scorer = "Unknown"
        if lineups:
            home_players = [p['player']['name'] for p in lineups[0]['startXI']] if lineups else []
            top_scorer = home_players[0] if home_players else "Unknown"

        predictions.append({
            'Match': f"{home_team} vs {away_team}",
            'Predicted Winner': le.inverse_transform([winner_pred])[0],
            'Double Chance': double_chance_pred,
            'Total Goals': round(goals_pred, 1),
            'Over/Under 2.5': '(Over 2.5)' if goals_pred > 2.5 else '(Under 2.5)',
            'Total Corners': round(corners_pred, 1),
            'Over/Under Corners 5.5': '(Over 5.5)' if corners_pred > 5.5 else '(Under 5.5)',
            'Total Cards': round(cards_pred, 1),
            'Over/Under Cards 3.5': '(Over 3.5)' if cards_pred > 3.5 else '(Under 3.5)',
            'First Half Goals': round(first_half_goals_pred, 1),
            'Second Half Goals': round(second_half_goals_pred, 1),
            'Asian Handicap': f"Home {round(goal_diff_pred, 1)}" if goal_diff_pred > 0 else f"Away {round(-goal_diff_pred, 1)}",
            'Clean Sheet (Home)': 'Yes' if clean_sheet_pred == 1 else 'No',
            'First Goal Scorer': top_scorer,
            'Exact Score': f"{int(round(goals_pred/2))}:{int(round(goals_pred/2))}"
        })

    return pd.DataFrame(predictions)

# Display predictions
if st.button("Get Predictions"):
    with st.spinner("Fetching data and predicting..."):
        pred_df = get_predictions(league_id, season)
        if not pred_df.empty:
            st.subheader("Predictions for All Betting Markets")
            st.dataframe(pred_df)
        else:
            st.warning("No upcoming matches found.")