from datetime import datetime, timezone

from sqlalchemy import create_engine, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, Session
)

import config


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Coach(Base):
    __tablename__ = "coaches"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))

    role: Mapped[str] = mapped_column(String(32), default="academy")
    sessions: Mapped[list["AnalysisSession"]] = relationship(back_populates="coach")


class AnalysisSession(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    coach_id: Mapped[int | None] = mapped_column(ForeignKey("coaches.id"), nullable=True)
    match_id: Mapped[int] = mapped_column(Integer)
    team: Mapped[str] = mapped_column(String(120))
    focus_problem: Mapped[str] = mapped_column(String(255), default="")
    overview: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    coach: Mapped["Coach"] = relationship(back_populates="sessions")
    suggestions: Mapped[list["Suggestion"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Suggestion(Base):
    __tablename__ = "suggestions"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    idea: Mapped[str] = mapped_column(Text)
    rationale: Mapped[str] = mapped_column(Text, default="")
    trade_offs: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[str] = mapped_column(String(16), default="medium")

    session: Mapped["AnalysisSession"] = relationship(back_populates="suggestions")
    outcome: Mapped["Outcome"] = relationship(
        back_populates="suggestion", uselist=False, cascade="all, delete-orphan"
    )


class Outcome(Base):
    __tablename__ = "outcomes"
    id: Mapped[int] = mapped_column(primary_key=True)
    suggestion_id: Mapped[int] = mapped_column(ForeignKey("suggestions.id"))
    applied: Mapped[str] = mapped_column(String(16), default="unknown")
    result: Mapped[str] = mapped_column(Text, default="")
    scored_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    suggestion: Mapped["Suggestion"] = relationship(back_populates="outcome")


_engine = create_engine(config.DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def init_db() -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(_engine)


def get_session() -> Session:
    return SessionLocal()


def save_analysis(match_id: int, team: str, recommendation: dict,
                  overview: str = "", coach_id: int | None = None) -> int:
    """Persist a coaching session + its suggestions. Returns the session id."""
    init_db()
    with get_session() as db:
        session = AnalysisSession(
            coach_id=coach_id,
            match_id=match_id,
            team=team,
            focus_problem=recommendation.get("focus_problem", ""),
            overview=overview,
        )
        for opt in recommendation.get("options", []):
            session.suggestions.append(Suggestion(
                idea=opt.get("idea", ""),
                rationale=opt.get("rationale", ""),
                trade_offs=opt.get("trade_offs", ""),
                evidence=opt.get("evidence", ""),
                confidence=opt.get("confidence", "medium"),
            ))
        db.add(session)
        db.commit()
        return session.id


def get_suggestions(session_id: int) -> list[dict]:
    """Return the stored suggestions for a session (with their DB ids)."""
    init_db()
    with get_session() as db:
        session = db.get(AnalysisSession, session_id)
        if session is None:
            return []
        return [
            {"id": s.id, "idea": s.idea, "confidence": s.confidence}
            for s in session.suggestions
        ]


def list_sessions(limit: int = 10) -> list[dict]:
    """Return recent sessions for a history view."""
    init_db()
    with get_session() as db:
        rows = (
            db.query(AnalysisSession)
            .order_by(AnalysisSession.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {"id": r.id, "match_id": r.match_id, "team": r.team,
             "focus_problem": r.focus_problem,
             "created_at": r.created_at.isoformat()}
            for r in rows
        ]


def record_outcome(suggestion_id: int, applied: str, result: str) -> int:
    """Record what actually happened after a suggestion - closes the DI loop."""
    init_db()
    with get_session() as db:
        outcome = Outcome(suggestion_id=suggestion_id, applied=applied, result=result)
        db.add(outcome)
        db.commit()
        return outcome.id


if __name__ == "__main__":
    init_db()
    print(f"Initialised database at: {config.DATABASE_URL}")
    demo = {
        "focus_problem": "Low conversion rate",
        "options": [
            {"idea": "Finishing drills", "confidence": "high",
             "evidence": "16 shots, 0 goals, xG 0.92"},
        ],
    }
    sid = save_analysis(22912, "Tottenham Hotspur", demo, overview="Demo session")
    print(f"Saved demo session id={sid}")
