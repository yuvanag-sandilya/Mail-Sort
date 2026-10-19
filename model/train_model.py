"""
Train a Logistic Regression model to classify email priority.
Generates a synthetic labeled dataset, trains the model, and saves it to disk.
"""

import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Synthetic training data ---
# Each tuple: (email_text, priority_label)
# Labels: 0 = Low, 1 = Normal, 2 = Important, 3 = Urgent

emails = [
    # ---- URGENT (3) ----
    ("URGENT: Server is down, all services are offline. Need immediate fix.", 3),
    ("CRITICAL: Production database crashed. Data loss possible. Act NOW.", 3),
    ("Emergency meeting in 10 minutes. CEO needs everyone in the boardroom.", 3),
    ("URGENT: Client threatening to cancel contract if not resolved today.", 3),
    ("System breach detected. Security incident requires immediate response.", 3),
    ("ASAP: Deployment failed in production. Rolling back but need help now.", 3),
    ("P0 incident: Payment processing is completely broken for all users.", 3),
    ("URGENT: Legal deadline is in 2 hours, need signature immediately.", 3),
    ("Fire alarm triggered in building B. Evacuate immediately.", 3),
    ("CRITICAL BUG: Users are being charged twice. Fix needed right now.", 3),
    ("Immediate action required: your account has been compromised.", 3),
    ("URGENT: Flight cancelled, need to rebook for the client dinner tonight.", 3),
    ("Server load at 99%, application will crash any moment. Need to scale NOW.", 3),
    ("EMERGENCY: Key employee just resigned, need to redistribute work today.", 3),
    ("Deadline in 1 hour: submit the compliance report or face penalties.", 3),
    ("URGENT: The investor demo is in 30 minutes and the app is broken.", 3),
    ("CRITICAL: Email system is down company-wide. No one can send or receive.", 3),
    ("Need your approval RIGHT NOW or we lose the deal.", 3),
    ("URGENT: Customer data exposed. Need to notify affected users immediately.", 3),
    ("Production is on fire. All hands on deck. Join the war room NOW.", 3),
    ("ASAP: The board presentation has errors, meeting starts in 20 minutes.", 3),
    ("URGENT: Payroll processing failed. Employees won't get paid on time.", 3),
    ("Critical vulnerability found in our system. Patch must be applied today.", 3),
    ("EMERGENCY: Office flooding on 3rd floor. Facilities team needed now.", 3),
    ("Drop everything: major client escalation, VP wants resolution within the hour.", 3),
    ("URGENT: API keys leaked on GitHub. Rotate all credentials NOW.", 3),
    ("The website is returning 500 errors for all users. Fix this immediately.", 3),
    ("CRITICAL: Ransomware detected on the network. Disconnect and respond now.", 3),
    ("ASAP: Client demo crashed live. They are still in the room waiting.", 3),
    ("Immediate escalation: SLA breach in progress, penalty clock is ticking.", 3),
    ("RED ALERT: Cloud bill spiked to $50k overnight. Shut down rogue instances NOW.", 3),
    ("URGENT: Data center power failure. Failover did not trigger. Manual intervention needed.", 3),
    ("All production containers are OOM killed. Services are completely offline.", 3),
    ("EMERGENCY: Main database replication lag is 3 hours. Risk of data loss.", 3),
    ("CRITICAL: SSL certificates expired. All HTTPS traffic is failing.", 3),
    ("Drop what you are doing. Board meeting moved to right now.", 3),
    ("P0: Authentication service is down. No user can log in.", 3),
    ("URGENT: Fraudulent transactions detected. Freeze the payment gateway immediately.", 3),
    ("EMERGENCY: Build pipeline is deploying broken code to production continuously.", 3),

    # ---- IMPORTANT (2) ----
    ("Reminder: Project proposal is due by end of day tomorrow.", 2),
    ("Please review the attached contract before our meeting on Thursday.", 2),
    ("The quarterly budget report needs your approval by Friday.", 2),
    ("Action needed: Update your security credentials before they expire next week.", 2),
    ("Meeting scheduled for tomorrow at 2 PM to discuss the product roadmap.", 2),
    ("Please complete your performance review by end of this week.", 2),
    ("The client requested changes to the proposal. Can you revise by Wednesday?", 2),
    ("Your pull request has merge conflicts that need to be resolved.", 2),
    ("Deadline approaching: Submit travel expenses before month end.", 2),
    ("Important: New company policy takes effect next Monday. Please read.", 2),
    ("Team lead meeting rescheduled to Thursday 3 PM. Attendance required.", 2),
    ("Need your input on the hiring decision. Please respond by tomorrow.", 2),
    ("The sprint retrospective is tomorrow. Please prepare your feedback.", 2),
    ("Contract renewal coming up next week. Review the terms and confirm.", 2),
    ("Please update the project timeline, stakeholders are asking for status.", 2),
    ("Your code review is blocking the release. Please review today if possible.", 2),
    ("Follow up on yesterday's discussion: we need to finalize the architecture.", 2),
    ("Action item from today's standup: document the API changes by Thursday.", 2),
    ("Training session on new tools scheduled for next Tuesday. Please register.", 2),
    ("The design mockups are ready for review. Feedback needed by Friday.", 2),
    ("Please confirm your attendance for the offsite next month.", 2),
    ("Budget approval needed for the new cloud infrastructure by end of week.", 2),
    ("Don't forget: 1-on-1 with your manager tomorrow morning.", 2),
    ("Your OKRs for next quarter are due this Friday.", 2),
    ("Important update to the deployment process — please read before your next release.", 2),
    ("Please coordinate with the QA team on the test plan for the new feature.", 2),
    ("Required: Sign-off for the new project budget before end of week.", 2),
    ("Action needed: Review the proposed security changes for the web app.", 2),
    ("Project timeline update — milestones have been adjusted, please review.", 2),
    ("Request for feedback on the new API design draft.", 2),
    ("Your input on the hiring strategy for Q4 is due by Tuesday.", 2),
    ("Please complete the compliance self-assessment by Friday.", 2),
    ("Important: Changes to the data privacy policy — please acknowledge receipt.", 2),
    ("Meeting with the product team on Monday to refine user stories.", 2),
    ("Documentation for the new microservice is overdue. Please complete.", 2),
    ("Action required: Reset your security token by EOD to avoid lockouts.", 2),
    ("Performance review input needed. Submit by the end of tomorrow.", 2),
    ("The launch checklist for next week is ready for your review.", 2),
    ("Contract review for new vendor required by Thursday morning.", 2),

    # ---- NORMAL (1) ----
    ("Hey, just wanted to check in on the progress of the feature.", 1),
    ("Here are the meeting notes from today's standup.", 1),
    ("Sharing the slides from yesterday's tech talk for reference.", 1),
    ("Can we schedule a call sometime this week to discuss the roadmap?", 1),
    ("FYI: I pushed a small refactor to the utils module.", 1),
    ("Attached is the report you asked about. Let me know if you have questions.", 1),
    ("Thanks for your help with the deployment yesterday.", 1),
    ("Quick question: do you know which branch has the latest changes?", 1),
    ("The test suite is passing now after the fix. Just a heads up.", 1),
    ("Sharing some articles I found relevant to our current project.", 1),
    ("Updated the wiki page with the new onboarding steps.", 1),
    ("Hey, are you free for a quick coffee chat this afternoon?", 1),
    ("Just an update: the vendor confirmed the delivery for next week.", 1),
    ("Here's the summary of the user feedback from last sprint.", 1),
    ("I'll be working from home tomorrow, reachable on Slack.", 1),
    ("Synced with the design team, they're on track for next milestone.", 1),
    ("Passing along this info from the support team about a feature request.", 1),
    ("Can you take a look at this PR when you get a chance? No rush.", 1),
    ("Lunch plans today? A few of us are heading to the new place downtown.", 1),
    ("Wanted to share my notes from the conference session I attended.", 1),
    ("Following up on our conversation last week about the migration plan.", 1),
    ("The staging environment is back up. All tests green.", 1),
    ("Here's the doc I mentioned during standup. Take a look when free.", 1),
    ("Just a heads up, I'll be out of office next Friday.", 1),
    ("Circling back on the API design. I've updated the draft — no rush to review.", 1),

    # ---- LOW (0) ----
    ("Weekly newsletter: Top 10 productivity tips for developers.", 0),
    ("You're invited to our annual company picnic! Details inside.", 0),
    ("New blog post on our engineering site: Lessons from scaling our platform.", 0),
    ("Office lunch menu for next week is now available.", 0),
    ("Reminder: The office will be closed on Monday for the holiday.", 0),
    ("Check out the new merch in the company store!", 0),
    ("Fun fact Friday: Did you know our codebase has over 2 million lines?", 0),
    ("Monthly book club: This month we're reading 'Designing Data-Intensive Apps'.", 0),
    ("Parking lot maintenance scheduled for this weekend. No action needed.", 0),
    ("Here's a cute photo of the office dog. Thought you'd enjoy it.", 0),
    ("Unsubscribe from this mailing list if you no longer wish to receive updates.", 0),
    ("Company softball game this Saturday. Come cheer the team on!", 0),
    ("FYI: The vending machine on floor 2 has been restocked.", 0),
    ("Happy birthday to everyone celebrating this month! 🎂", 0),
    ("New wallpapers available for your work laptop. Download here.", 0),
    ("Just for fun: What's your favorite IDE theme? Vote in our poll.", 0),
    ("The recycling bins have been moved to the kitchen area.", 0),
    ("Throwback Thursday: Photos from last year's hackathon.", 0),
    ("Free webinar this Friday: Introduction to Kubernetes. Optional.", 0),
    ("The gym on floor 1 has new equipment. Feel free to try it out.", 0),
    ("Lost and found: Someone left a blue umbrella in the conference room.", 0),
    ("Our team playlist on Spotify has been updated. Give it a listen.", 0),
    ("Casual Friday reminder: Dress code is relaxed tomorrow.", 0),
    ("Just sharing: Funny meme about debugging in production.", 0),
    ("The new coffee machine is here! It's in the break room on floor 3.", 0),
]

def train():
    texts = [e[0] for e in emails]
    labels = [e[1] for e in emails]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        C=1.0
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Low", "Normal", "Important", "Urgent"]
    ))

    # Save model and vectorizer
    save_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(save_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(save_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Model and vectorizer saved to {save_dir}")

if __name__ == "__main__":
    train()
