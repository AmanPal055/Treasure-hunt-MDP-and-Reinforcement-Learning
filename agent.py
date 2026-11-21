"""
COL333 Assignment 3 - Game Playing AI
Phase 2 Tournament Agent
Team: Aman (2026MCS2258), Anshul (2026MCS2655)
"""

import math
from pyrusgeom.geom_2d import Sector2D, Vector2D
from src.student.interface import StarterAgent, Action, Tools, note


def clamp(val: float, low: float, high: float) -> float:
    if val < low:
        return low
    if val > high:
        return high
    return val


def norm_angle(ang: float) -> float:
    while ang > 180.0:
        ang -= 360.0
    while ang < -180.0:
        ang += 360.0
    return ang


def seg_point_dist(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-6:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / l2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(ax + t * dx - px, ay + t * dy - py)


class Agent1(StarterAgent):
    def __init__(self):
        super().__init__()
        # State utility evaluation weights
        self.w_poss = 3.4
        self.w_goal = 3.8
        self.w_prog = 2.6
        self.w_radial = 2.4
        self.w_space = 1.25
        self.w_risk = 3.2

        # Geometry & distance thresholds
        self.shoot_range = 32.5
        self.post_margin = 0.80
        self.gk_catch_radius = 1.15
        self.kickable_radius = 1.05

        # Passing parameters
        self.pass_radius = 2.8
        self.pass_lead_radius = 6.5
        self.pass_samples = 8
        self.pass_max_dist = 36.0
        self.pass_min_dist = 2.5
        self.pass_corridor_deg = 9.0

        # Dribble parameters
        self.dribble_dist = 4.5
        self.dribble_angle = 24.0

        # Expectimax pass evaluation settings
        self.intercept_softness = 0.84
        self.max_near_opps = 4
        self.min_decision_ms = 75

        # Persistent state and action commitment
        self._mem = {}
        self.hysteresis_bonus = 0.22
        self.kick_commit_bonus = 0.85
        self.early_flight_veto = True
        self.veto_cycles = 2

    # ------------------------------------------------------------------
    # Helper math & lookups
    # ------------------------------------------------------------------
    def _get_valid_teammates(self, wm):
        return [t for t in wm.teammates if t.uniform_number > 0 and not t.is_goalie]

    def _get_valid_opponents(self, wm):
        return [o for o in wm.opponents if o.uniform_number > 0]

    def _cycles_for_distance(self, agent, player, dist: float) -> int:
        if dist <= 0.05:
            return 0
        try:
            return Tools.cycles_to_reach_distance(agent, player, dist)
        except Exception:
            speed = getattr(agent.server_params, "default_player_speed_max", 1.0)
            return max(0, int(math.ceil(dist / max(speed, 0.1))))

    def _turn_penalty(self, player, point: Vector2D, raw_dist: float, ctrl_radius: float) -> int:
        # Estimate turning cycles required before dashing toward a point
        if raw_dist <= ctrl_radius + 0.15:
            return 0
        try:
            target_ang = (point - player.position).th().degree()
            body_ang = float(player.body_direction)
            diff = abs(norm_angle(target_ang - body_ang))
            cone = max(18.0, math.degrees(math.asin(min(1.0, ctrl_radius / max(raw_dist, 0.1)))))
            rem = diff - cone
            if rem <= 15.0:
                return 0
            if rem <= 85.0:
                return 1
            return 2
        except Exception:
            return 0

    def _cycles_to_point(self, agent, player, point: Vector2D, is_opp: bool = False) -> int:
        # Reach cycle estimate factoring in turn delay and control radius
        sp = agent.server_params
        raw_dist = player.position.dist(point)
        if getattr(player, "is_goalie", False) and is_opp:
            ctrl_r = max(getattr(sp, "catch_area_l", 0.0), getattr(sp, "catchable_area", 0.0), self.gk_catch_radius)
        elif is_opp:
            ctrl_r = getattr(sp, "kickable_area", self.kickable_radius)
        else:
            ctrl_r = getattr(sp, "kickable_area", self.kickable_radius) * 0.84

        if raw_dist <= ctrl_r:
            return 0
        dash_dist = max(0.0, raw_dist - ctrl_r)
        dash_c = self._cycles_for_distance(agent, player, dash_dist)
        turn_c = self._turn_penalty(player, point, raw_dist, ctrl_r)
        return dash_c + turn_c

    def _estimate_pass_steps(self, agent, dist: float) -> int:
        # Calibration for pass pace (2.38 - 2.84 m/s) to beat close-down
        sp = agent.server_params
        v_target = 2.84 if dist > 15.0 else (2.62 if dist > 8.5 else 2.38)
        v_pass = min(sp.ball_speed_max * 0.95, v_target)
        steps = Tools.steps_required(v_pass, dist, sp.ball_decay, sp.ball_speed_max)
        if steps is None or steps < 1:
            steps = max(1, int(dist / 2.1))
        return max(1, min(int(steps), 16))

    def _one_step_kick_speed(self, agent, target: Vector2D) -> float:
        # Max velocity achievable in a single touch toward target
        try:
            wm = agent.wm
            sp = agent.server_params
            ang = (target - wm.ball.position).th()
            bvel_next = wm.ball.velocity * sp.ball_decay
            vmax = Tools.calc_max_velocity(ang, wm.self.kick_rate, bvel_next, sp)
            return vmax.r()
        except Exception:
            return agent.server_params.ball_speed_max * 0.8

    # ------------------------------------------------------------------
    # Ball flight & Expectimax pass evaluation
    # ------------------------------------------------------------------
    def _flight_points(self, agent, ball: Vector2D, target: Vector2D, n: int):
        # Ball positions over discrete flight cycles k = 1..n
        d = agent.server_params.ball_decay
        denom = 1.0 - d ** n
        if denom < 1e-9:
            return [target]
        dx = target.x() - ball.x()
        dy = target.y() - ball.y()
        pts = []
        for k in range(1, n + 1):
            frac = (1.0 - d ** k) / denom
            pts.append(Vector2D(ball.x() + dx * frac, ball.y() + dy * frac))
        return pts

    def _intercept_lead(self, agent, ball: Vector2D, target: Vector2D, n: int, opps):
        # Peak opponent intercept lead over the ball flight
        best_lead, best_pt = -99.0, target
        for k, pt in enumerate(self._flight_points(agent, ball, target, n), 1):
            for o in opps:
                t_opp = self._cycles_to_point(agent, o, pt, is_opp=True)
                lead = k - t_opp
                if lead > best_lead:
                    best_lead, best_pt = lead, pt
        return best_lead, best_pt

    def _p_lost(self, lead: float) -> float:
        # Interception probability via logistic curve over lead margin
        if lead <= -2.5:
            return 0.0
        z = clamp((lead - 0.12) / self.intercept_softness, -20.0, 20.0)
        return 1.0 / (1.0 + math.exp(-z))

    def _lost_ball_value(self, agent, point: Vector2D) -> float:
        # Turnover evaluation when intercepted at point
        L = agent.server_params.pitch_half_length
        own_goal = Vector2D(-L, 0.0)
        danger = max(0.0, 1.0 - point.dist(own_goal) / (self.shoot_range + 16.0))
        return (-self.w_poss
                + self.w_prog * clamp(point.x() / L, -1.0, 1.0)
                - (self.w_goal + 1.0) * danger)

    def expectimax_pass_value(self, agent, ball: Vector2D, target: Vector2D,
                              arrive_in: int, t_rcv: int, t_opp: int, opps) -> float:
        # Receiver meets pass on run during OUR_PASS
        eff_t_our = min(max(arrive_in, t_rcv), t_rcv + 2)
        u_recv = self.evaluate_state(agent, target, eff_t_our, t_opp, target, opps, is_pass=True)
        if agent.budget.decision_remaining_ms < self.min_decision_ms:
            return u_recv
        near = sorted(opps, key=lambda o: o.position.dist(ball) + o.position.dist(target))[:self.max_near_opps]
        if not near:
            return u_recv
        lead, pt = self._intercept_lead(agent, ball, target, arrive_in, near)
        rcv_lead = float(t_rcv - t_opp)
        eff_lead = max(float(lead), rcv_lead)
        p = self._p_lost(eff_lead)
        return (1.0 - p) * u_recv + p * self._lost_ball_value(agent, pt)

    # ------------------------------------------------------------------
    # State evaluation
    # ------------------------------------------------------------------
    def evaluate_state(self, agent, ball_pos: Vector2D, t_our: int, t_opp: int,
                       target_pos: Vector2D, opps, is_pass: bool = False) -> float:
        # Multi-factor utility function
        sp = agent.server_params
        L = sp.pitch_half_length
        W = sp.pitch_half_width
        goal = Vector2D(L, 0.0)
        cur_ball = agent.wm.ball.position

        # Pitch progression along x-axis
        f_prog = clamp(ball_pos.x() / L, -1.0, 1.0)

        # Distance reduction toward the opponent goal
        f_radial = clamp((cur_ball.dist(goal) - ball_pos.dist(goal)) / 26.0, -1.0, 1.0)

        # Possession reach margin
        scale = 2.2 if is_pass else 4.5
        f_poss = clamp((t_opp - t_our) / scale, -1.0, 1.0)

        # Goal scoring proximity
        dist_to_goal = ball_pos.dist(goal)
        if dist_to_goal < self.shoot_range + 12.0:
            f_goal = (1.0 - (dist_to_goal / (self.shoot_range + 12.0))) ** 1.25
        else:
            f_goal = 0.0

        # Space around receiver from nearest opponent
        if opps:
            min_opp_dist = min(o.position.dist(target_pos) for o in opps)
            f_space = clamp(min_opp_dist / 8.5, 0.0, 1.0)
        else:
            f_space = 1.0

        # Central attacking funnel & avoiding getting pinned in wide corners
        f_center = 0.0
        if target_pos.x() > 10.0:
            y_norm = abs(target_pos.y()) / (W * 0.75)
            f_center = 0.65 * (1.0 - min(1.0, y_norm))
        if target_pos.x() > 26.0 and abs(target_pos.y()) < 9.0:
            f_center += 2.50
        elif target_pos.x() > 38.0 and abs(target_pos.y()) > 24.0:
            f_center -= 1.20

        # Defensive half turnover risk
        f_risk = 0.0
        if target_pos.x() < -L * 0.30 and abs(target_pos.y()) < W * 0.65:
            f_risk = clamp(((-L * 0.30) - target_pos.x()) / (L * 0.70), 0.0, 1.0)

        return (self.w_poss * f_poss +
                self.w_goal * f_goal +
                self.w_prog * f_prog +
                self.w_radial * f_radial +
                self.w_space * f_space +
                f_center -
                self.w_risk * f_risk)

    def _evaluate_ply2(self, agent, landing_pt: Vector2D, rcv_player, opps, teammates) -> float:
        # 2-step lookahead: evaluate receiver follow-up options
        sp = agent.server_params
        L = sp.pitch_half_length
        goal = Vector2D(L, 0.0)
        d_goal = landing_pt.dist(goal)

        # Immediate follow-up shot chance
        if d_goal <= 20.0 and abs(landing_pt.y()) < 11.5:
            return 4.5 * (1.0 - d_goal / 23.5)

        # Secondary progression or pass option
        best_secondary = 0.0
        for tm2 in teammates:
            if rcv_player is not None and tm2.uniform_number == rcv_player.uniform_number:
                continue
            if tm2.position.x() < landing_pt.x() - 2.0:
                continue
            f_prog2 = clamp(tm2.position.x() / L, -1.0, 1.0)
            min_opp = min((o.position.dist(tm2.position) for o in opps), default=8.0)
            f_space2 = clamp(min_opp / 8.0, 0.0, 1.0)
            u2 = 1.40 * f_prog2 + 0.85 * f_space2
            if u2 > best_secondary:
                best_secondary = u2

        return max(0.0, best_secondary)

    def _early_flight_blocked(self, agent, ball: Vector2D, target: Vector2D, arrive_in: int, opps) -> bool:
        # Check if an opponent can block the pass in the first veto_cycles
        n = max(1, arrive_in)
        for k, pt in enumerate(self._flight_points(agent, ball, target, n), 1):
            if k > self.veto_cycles:
                break
            for o in opps:
                if k - self._cycles_to_point(agent, o, pt, is_opp=True) >= 0:
                    return True
        return False

    def _offside_risk_on_pass(self, wm, ball: Vector2D, target: Vector2D, recv_tm, teammates) -> bool:
        # Check if receiver or teammate in corridor is offside at the moment of kick
        eff_line = max(0.0, float(wm.offside_line_x), ball.x())
        if recv_tm is not None and recv_tm.position.x() > eff_line - 0.30:
            return True
        for tm in teammates:
            if tm.position.x() > eff_line - 0.20:
                if seg_point_dist(ball.x(), ball.y(), target.x(), target.y(),
                                  tm.position.x(), tm.position.y()) < 4.2:
                    return True
        return False

    # ------------------------------------------------------------------
    # Action history & hysteresis
    # ------------------------------------------------------------------
    def _update_pass_streak(self, agent) -> int:
        mem = self._mem
        cyc = agent.wm.cycle
        if mem.get("last_ob_cycle") == cyc - 1 and mem.get("last_ob_type") in ('pass', 'space_pass'):
            streak = mem.get("pass_streak", 1) + 1
        else:
            streak = 1
        return streak

    def _update_dribble_streak(self, agent) -> int:
        mem = self._mem
        cyc = agent.wm.cycle
        if mem.get("last_ob_cycle") == cyc - 1 and mem.get("last_ob_type") == 'dribble':
            return mem.get("drib_streak", 0) + 1
        return 0

    def _update_hold_streak(self, agent) -> int:
        mem = self._mem
        cyc = agent.wm.cycle
        if mem.get("last_ob_cycle") == cyc - 1 and mem.get("last_ob_type") == 'hold':
            return mem.get("hold_streak", 0) + 1
        return 0

    def _remember_choice(self, agent, act_type: str, point: Vector2D, p_streak: int, d_streak: int, h_streak: int = 0):
        mem = self._mem
        mem["last_ob_cycle"] = agent.wm.cycle
        mem["last_ob_type"] = act_type
        mem["hy_target"] = point
        mem["pass_streak"] = p_streak if act_type in ('pass', 'space_pass') else 0
        mem["drib_streak"] = d_streak if act_type == 'dribble' else 0
        mem["hold_streak"] = h_streak if act_type == 'hold' else 0
        if p_streak >= 4:
            note(agent, "pass_stalled")

    def _apply_hysteresis(self, agent, candidates, cur_pass_streak: int):
        mem = self._mem
        if mem.get("last_ob_cycle") != agent.wm.cycle - 1:
            return candidates
        last_type = mem.get("last_ob_type")
        last_pt = mem.get("hy_target")
        if last_type not in ('pass', 'space_pass') or last_pt is None:
            return candidates

        bonus = self.kick_commit_bonus if cur_pass_streak == 2 else self.hysteresis_bonus
        out = []
        for c in candidates:
            score = c[2]
            if c[0] == last_type and c[3] is not None and c[3].dist(last_pt) < 2.5:
                score += bonus
                out.append((c[0], last_pt, score, c[3], c[4], c[5], c[6]))
            else:
                out.append(c)
        return out

    # ------------------------------------------------------------------
    # On-ball decision making
    # ------------------------------------------------------------------
    def on_ball(self, agent) -> None:
        try:
            self._on_ball_impl(agent)
        except Exception as e:
            note(agent, f"err_{type(e).__name__}")
            Action.clear(agent)

    def _on_ball_impl(self, agent) -> None:
        wm = agent.wm
        sp = agent.server_params
        ball = wm.ball.position
        L = sp.pitch_half_length
        W = sp.pitch_half_width
        goal_w = sp.goal_width

        teammates = self._get_valid_teammates(wm)
        opps = self._get_valid_opponents(wm)
        outfield_opps = [o for o in opps if not o.is_goalie]

        nearest_opp_dist = min((o.position.dist(ball) for o in opps), default=100.0)
        t_opp_tackle = min((self._cycles_to_point(agent, o, ball, is_opp=True) for o in opps), default=100)
        pressed = (nearest_opp_dist < 3.2 or t_opp_tackle <= 2)
        severe_danger = (ball.x() < -22.0 and (nearest_opp_dist < 1.85 or t_opp_tackle <= 0))

        cur_pass_streak = self._update_pass_streak(agent)
        cur_drib_streak = self._update_dribble_streak(agent)
        cur_hold_streak = self._update_hold_streak(agent)
        stalling = (cur_pass_streak >= 3)

        if severe_danger:
            note(agent, "clear_emergency")
            self._execute_safe_clear(agent, ball, L, W, True)
            return

        candidates = []

        # 1. Shot evaluation
        dist_to_goal_center = ball.dist(Vector2D(L, 0.0))
        if dist_to_goal_center <= self.shoot_range and abs(ball.y()) < 24.0:
            gk = Tools.get_opponent_goalie(agent)
            half_g = goal_w / 2.0 - self.post_margin
            fractions = (-1.0, -0.85, -0.70, -0.50, 0.50, 0.70, 0.85, 1.0)
            shot_targets = [Vector2D(L, f * half_g) for f in fractions]

            for post in shot_targets:
                shot_dist = ball.dist(post)
                if shot_dist > 32.0:
                    continue

                dx = post.x() - ball.x()
                dy = post.y() - ball.y()
                seg_len2 = dx * dx + dy * dy
                if seg_len2 < 1e-4:
                    continue

                one_step_v = self._one_step_kick_speed(agent, post)
                if one_step_v < 1.40:
                    continue
                n_ball = Tools.ball_move_step(one_step_v, shot_dist, sp.ball_decay)
                if n_ball is None or n_ball < 1 or n_ball > 16:
                    continue

                # Outfield defender block check
                blocked_by_outfield = False
                for o in outfield_opps:
                    if o.position.x() <= ball.x() + 0.4:
                        continue
                    t_proj = ((o.position.x() - ball.x()) * dx + (o.position.y() - ball.y()) * dy) / seg_len2
                    if t_proj <= 0.02 or t_proj >= 0.98:
                        continue
                    proj_pt = Vector2D(ball.x() + t_proj * dx, ball.y() + t_proj * dy)
                    perp_d = o.position.dist(proj_pt)
                    if perp_d > 1.85:
                        continue
                    d_proj = ball.dist(proj_pt)
                    n_proj = Tools.ball_move_step(one_step_v, d_proj, sp.ball_decay)
                    if n_proj is None or n_proj < 1:
                        n_proj = 1
                    n_opp = self._cycles_to_point(agent, o, proj_pt, is_opp=True)
                    if n_opp <= n_proj:
                        blocked_by_outfield = True
                        break

                if blocked_by_outfield and dist_to_goal_center > 17.5:
                    continue

                # Goalkeeper save check over trajectory
                flight_pts = self._flight_points(agent, ball, post, max(1, n_ball))
                gk_line_dist = 9.0
                if gk is not None:
                    gk_line_dist = seg_point_dist(ball.x(), ball.y(), post.x(), post.y(),
                                                  gk.position.x(), gk.position.y())
                    react_delay = 1 if shot_dist <= 22.0 else 0
                    dist_gk_post = max(0.0, gk.position.dist(post) - 1.25)
                    t_gk_post = self._cycles_for_distance(agent, gk, dist_gk_post) + react_delay
                    post_beat_margin = t_gk_post - n_ball
                    gk_post_dy = abs(gk.position.y() - post.y())

                    min_gk_margin = 99
                    for k_step, pt in enumerate(flight_pts, 1):
                        if pt.x() >= L - 1.2 or k_step >= n_ball:
                            break
                        n_gk = self._cycles_to_point(agent, gk, pt, is_opp=True) + react_delay
                        margin = n_gk - k_step
                        if margin < min_gk_margin:
                            min_gk_margin = margin
                    gk_margin = min_gk_margin
                else:
                    gk_line_dist = 9.0
                    gk_margin = 5
                    post_beat_margin = 5
                    gk_post_dy = 8.0

                shot_score = -9.0
                if gk_line_dist < 1.45:
                    continue
                if gk is not None and gk.position.dist(ball) < 1.75:
                    continue
                if gk_margin < 0 and gk_line_dist < 2.5 and dist_to_goal_center > 11.0:
                    continue

                # Base scoring tiers
                if gk_margin >= 1 and not blocked_by_outfield:
                    shot_score = 12.0 + 1.2 * clamp(gk_margin, 1, 4) + 0.35 * gk_line_dist
                elif dist_to_goal_center <= 22.0 and ball.x() >= 30.0 and (gk_margin >= 0 or (gk_line_dist >= 2.2 and gk_post_dy >= 2.5)) and gk_line_dist >= 1.55 and not blocked_by_outfield:
                    shot_score = 11.6 + 0.45 * gk_line_dist
                elif ball.x() >= 33.5 and abs(ball.y()) <= 10.0 and (gk_margin >= 0 or (gk_line_dist >= 2.0 and gk_post_dy >= 2.5)) and (gk_post_dy >= 2.5 or gk_line_dist >= 1.6) and not blocked_by_outfield:
                    shot_score = 11.4 + 0.40 * gk_line_dist
                elif shot_dist <= 32.0 and (gk_margin >= 0 or (gk_line_dist >= 2.4 and gk_post_dy >= 2.8)) and (gk_post_dy >= 2.8 or gk_line_dist >= 1.9) and not blocked_by_outfield:
                    shot_score = 11.0 + 0.35 * gk_line_dist
                elif (shot_dist <= 32.0 and ball.x() >= 20.0 and nearest_opp_dist > 2.5 and
                      gk is not None and gk_post_dy >= 2.8 and gk_line_dist >= 1.8 and not blocked_by_outfield):
                    shot_score = 11.2 + 0.35 * gk_line_dist
                elif dist_to_goal_center <= 14.0 and gk_line_dist >= 1.2 and not blocked_by_outfield:
                    shot_score = 10.5 + 0.40 * gk_line_dist

                # Distance arrival penalty
                shot_score -= 0.15 * n_ball

                # Lateral separation & far post bonus
                shot_score += 0.35 * gk_post_dy
                if ball.y() * post.y() < -1.0:
                    if gk_post_dy >= 4.0:
                        shot_score += 1.50
                    elif gk_post_dy >= 2.5:
                        shot_score += 0.80

                # Corner placement bonus
                if abs(post.y()) >= 5.0:
                    shot_score += 1.10
                elif abs(post.y()) >= 4.0:
                    shot_score += 0.50

                if nearest_opp_dist < 2.5 or dist_to_goal_center <= 16.0:
                    shot_score += 1.25
                else:
                    shot_score += 0.45

                if shot_score >= 9.5:
                    candidates.append(('shot', post, shot_score, post, None, 1, True))

        # 2. Pass candidates
        eff_offside_line = max(0.0, float(wm.offside_line_x), ball.x())
        our_gk = next((t for t in wm.teammates if t.is_goalie), None)

        for tm in teammates:
            tm_is_onside = (tm.position.x() <= eff_offside_line - 0.30)
            if not tm_is_onside and tm.position.x() > 0.0:
                continue

            targets = [tm.position] + Tools.points_around(tm.position, self.pass_radius, self.pass_samples)
            for lp in Tools.points_around(tm.position, self.pass_lead_radius, 6):
                if lp.x() >= tm.position.x() - 1.0:
                    targets.append(lp)
            if tm.velocity.r() > 0.15:
                targets.append(tm.position + tm.velocity * 3.0)

            # Through balls into space
            if tm_is_onside and tm.position.x() > -18.0:
                goal_aim_y = clamp(tm.position.y() * 0.45, -7.5, 7.5)
                goal_dir = (Vector2D(L, goal_aim_y) - tm.position).th().degree()
                for lead_d in (6.5, 11.0, 15.5):
                    tb_pt = tm.position + Vector2D.polar2vector(lead_d, goal_dir)
                    if tb_pt.x() <= L - 3.5:
                        targets.append(tb_pt)

            # Cutback targets from wide positions into box
            if ball.x() > 12.0 and abs(ball.y()) > 3.5 and tm_is_onside:
                for cut_x in (34.0, 38.0, 42.0):
                    for cut_y in (-5.0, -2.5, 0.0, 2.5, 5.0):
                        cut_pt = Vector2D(cut_x, cut_y)
                        if cut_pt.dist(tm.position) < 11.5:
                            targets.append(cut_pt)

            for target in targets:
                if abs(target.x()) > L - 1.5 or abs(target.y()) > W - 1.3:
                    continue
                p_dist = ball.dist(target)
                max_allowed_dist = 36.0 if ball.x() < -18.0 else self.pass_max_dist
                if p_dist < self.pass_min_dist or p_dist > max_allowed_dist:
                    continue

                # Defensive box backpass protection
                if target.x() < -34.0:
                    min_opp_to_target = min((o.position.dist(target) for o in opps), default=100.0)
                    if min_opp_to_target < 8.5:
                        continue
                if our_gk is not None and target.dist(our_gk.position) < 8.0:
                    min_opp_to_gk = min((o.position.dist(our_gk.position) for o in opps), default=100.0)
                    if min_opp_to_gk < 9.0:
                        continue

                if self._offside_risk_on_pass(wm, ball, target, tm, teammates):
                    continue

                pass_angle = (target - ball).th().degree()
                lane = Sector2D(ball, 0.7, p_dist + 0.8,
                                pass_angle - self.pass_corridor_deg,
                                pass_angle + self.pass_corridor_deg)
                if any(lane.contains(o.position) for o in outfield_opps):
                    continue

                arrive_in = self._estimate_pass_steps(agent, p_dist)
                if self.early_flight_veto and self._early_flight_blocked(agent, ball, target, arrive_in, opps):
                    continue

                t_rcv = self._cycles_to_point(agent, tm, target, is_opp=False)
                if target.x() > 22.0:
                    t_opp_outfield = min((self._cycles_to_point(agent, o, target, is_opp=True) for o in outfield_opps), default=100)
                    t_gk = min((self._cycles_to_point(agent, o, target, is_opp=True) for o in opps if getattr(o, "is_goalie", False)), default=100)
                    if t_rcv > t_opp_outfield or t_rcv > t_gk + 1 or arrive_in > t_opp_outfield + 1:
                        continue
                    t_opp = min(t_opp_outfield, t_gk)
                else:
                    t_opp = min((self._cycles_to_point(agent, o, target, is_opp=True) for o in opps), default=100)
                    if t_rcv + 1 > t_opp or arrive_in > t_opp + 1:
                        continue

                score = self.expectimax_pass_value(agent, ball, target, arrive_in, t_rcv, t_opp, opps)

                # Progression & territory bonuses
                gain = target.x() - ball.x()
                if gain > 3.0 and (t_opp - t_rcv) >= 0:
                    score += 1.35 * clamp(gain / 16.0, 0.0, 1.2)
                if gain > 9.0 and target.x() > -2.0 and (t_opp - t_rcv) >= 0:
                    score += 1.85
                if target.x() > 14.0 and (t_opp - t_rcv) >= 0:
                    score += 1.50
                if target.x() > eff_offside_line + 1.0 and (t_opp - t_rcv) >= 1:
                    score += 1.80
                if target.x() > 30.0 and abs(target.y()) < 11.5:
                    score += 2.80

                # Backpass penalties to avoid giving away field position
                if ball.x() > 18.0 and gain < -2.5:
                    score -= 4.50
                elif ball.x() >= 0.0 and gain < -3.0:
                    score -= 3.50
                elif ball.x() < 0.0 and gain < -5.0 and not pressed:
                    score -= 1.80

                accept_late_val = 5 if pressed else 4
                candidates.append(('pass', target, score, target, tm, arrive_in, accept_late_val))

        # 3. Space-pass candidates
        if ball.x() < -10.0:
            space_distances = (10.0, 18.0, 26.0, 32.0)
        else:
            space_distances = (9.0, 15.0, 22.0, 28.0)
        space_angles = (-30.0, -15.0, 0.0, 15.0, 30.0)
        for s_dist in space_distances:
            for s_ang in space_angles:
                if ball.y() > 12.0 and s_ang > 5.0:
                    continue
                if ball.y() < -12.0 and s_ang < -5.0:
                    continue
                target = ball + Vector2D.polar2vector(s_dist, s_ang)
                if abs(target.x()) > L - 2.5 or abs(target.y()) > W - 2.0:
                    continue
                if target.x() > L - 8.0:
                    continue
                if target.x() < -30.0:
                    continue
                if self._offside_risk_on_pass(wm, ball, target, None, teammates):
                    continue

                pass_angle = (target - ball).th().degree()
                lane = Sector2D(ball, 0.8, min(s_dist * 0.75, 16.0), pass_angle - 10.5, pass_angle + 10.5)
                if any(lane.contains(o.position) for o in outfield_opps):
                    continue

                arrive_in = self._estimate_pass_steps(agent, s_dist)
                if self.early_flight_veto and self._early_flight_blocked(agent, ball, target, arrive_in, opps):
                    continue

                onside_mates = [tm for tm in teammates if tm.position.x() <= eff_offside_line - 0.30]
                if not onside_mates:
                    continue
                best_tm = min(onside_mates, key=lambda tm: self._cycles_to_point(agent, tm, target, is_opp=False))
                t_tm = self._cycles_to_point(agent, best_tm, target, is_opp=False)
                if target.x() > 22.0:
                    t_opp_outfield = min((self._cycles_to_point(agent, o, target, is_opp=True) for o in outfield_opps), default=100)
                    t_gk = min((self._cycles_to_point(agent, o, target, is_opp=True) for o in opps if getattr(o, "is_goalie", False)), default=100)
                    if t_tm > t_opp_outfield or t_tm > t_gk + 1 or arrive_in > t_opp_outfield + 1:
                        continue
                    t_opp = min(t_opp_outfield, t_gk)
                else:
                    t_opp = min((self._cycles_to_point(agent, o, target, is_opp=True) for o in opps), default=100)
                    if t_tm + 1 > t_opp or arrive_in > t_opp + 1:
                        continue
                score = self.expectimax_pass_value(agent, ball, target, arrive_in, t_tm, t_opp, opps)

                gain = target.x() - ball.x()
                if gain > 3.0 and (t_opp - t_tm) >= 0:
                    score += 1.30 * clamp(gain / 16.0, 0.0, 1.2)
                if gain > 9.0 and target.x() > -2.0 and (t_opp - t_tm) >= 0:
                    score += 1.85
                if target.x() > 14.0 and (t_opp - t_tm) >= 0:
                    score += 1.50
                if target.x() > eff_offside_line + 1.0 and (t_opp - t_tm) >= 1:
                    score += 1.80
                if target.x() > 30.0 and abs(target.y()) < 11.5:
                    score += 2.80

                if ball.x() > 18.0 and gain < -2.5:
                    score -= 4.50
                elif ball.x() >= 0.0 and gain < -3.0:
                    score -= 3.50
                elif ball.x() < 0.0 and gain < -5.0 and not pressed:
                    score -= 1.80

                accept_late_val = 5 if pressed else 4
                candidates.append(('space_pass', target, score, target, best_tm, arrive_in, accept_late_val))

        # 4. Dribble candidates
        can_dribble = (nearest_opp_dist > 1.75)
        if can_dribble:
            aim_x = 38.0 if (ball.x() > 20.0 and abs(ball.y()) > 6.0) else L
            goal_vec = Vector2D(aim_x, 0.0)
            base_angle = (goal_vec - ball).th().degree()
            drib_dists = (2.4, 4.5) if ball.x() > 10.0 else (self.dribble_dist,)
            for d_dist in drib_dists:
                for off_deg in (-60.0, -38.0, -self.dribble_angle, 0.0, self.dribble_angle, 38.0, 60.0):
                    if abs(off_deg) > 42.0 and not pressed and ball.x() > 18.0:
                        continue
                    dribble_angle = base_angle + off_deg
                    target = ball + Vector2D.polar2vector(d_dist, dribble_angle)
                    if abs(target.x()) > L - 1.2 or abs(target.y()) > W - 1.2:
                        continue
                    if ball.x() < -18.0 and target.x() < ball.x() + 0.5:
                        continue

                    lane = Sector2D(ball, 0.4, d_dist + 1.2, dribble_angle - 16.0, dribble_angle + 16.0)
                    if any(lane.contains(o.position) for o in outfield_opps):
                        continue

                    mid_pt = Vector2D(0.5 * (ball.x() + target.x()), 0.5 * (ball.y() + target.y()))
                    t_our_mid = self._cycles_to_point(agent, wm.self, mid_pt, is_opp=False)
                    t_opp_mid = min((self._cycles_to_point(agent, o, mid_pt, is_opp=True) for o in opps), default=100)
                    if t_opp_mid <= t_our_mid:
                        continue

                    t_our = self._cycles_to_point(agent, wm.self, target, is_opp=False)
                    t_opp = min((self._cycles_to_point(agent, o, target, is_opp=True) for o in opps), default=100)
                    req_lead = 0 if d_dist <= 2.5 else 1
                    if t_opp <= t_our + req_lead:
                        continue

                    raw_drib = self.evaluate_state(agent, target, t_our, t_opp, target, opps, is_pass=False)
                    drib_lead = float(max(t_our - t_opp, t_our_mid - t_opp_mid))
                    p_drib_lost = self._p_lost(drib_lead + 0.6)
                    score = (1.0 - p_drib_lost) * raw_drib + p_drib_lost * self._lost_ball_value(agent, mid_pt)

                    if cur_drib_streak >= 1:
                        if ball.x() < 28.0:
                            score -= 0.32 * min(cur_drib_streak, 4)
                        elif cur_drib_streak >= 3:
                            score -= 0.40 * min(cur_drib_streak - 2, 4)
                    if ball.x() > 20.0 and target.x() > ball.x() + 0.9 and abs(target.y()) < 9.0:
                        score += 1.35
                    if ball.x() > 10.0 and abs(ball.y()) > 6.5 and abs(target.y()) < abs(ball.y()) - 0.7:
                        score += 0.90
                    if ball.x() > 0.0 and target.x() > ball.x() + 0.8:
                        score += 0.85
                    if stalling:
                        score += 0.70
                    candidates.append(('dribble', target, score, target, None, 0, 0))

        # 5. Hold candidate (only when unpressured in defensive half)
        if (cur_hold_streak == 0 and ball.x() < -8.0 and abs(ball.y()) < 14.0 and
                nearest_opp_dist > 6.0 and t_opp_tackle > 5):
            hold_score = self.evaluate_state(agent, ball, 0, t_opp_tackle, ball, opps, is_pass=False) - 3.2
            candidates.append(('hold', None, hold_score, ball, None, 0, 0))

        # 6. Action selection & optional 2-ply search
        if not candidates:
            note(agent, "clear_no_cand")
            self._execute_safe_clear(agent, ball, L, W, pressed)
            return

        candidates = self._apply_hysteresis(agent, candidates, cur_pass_streak)
        candidates.sort(key=lambda c: c[2], reverse=True)

        # Clear goal opportunity: shoot immediately if viable
        best_shot = max((c for c in candidates if c[0] == 'shot'), key=lambda c: c[2], default=None)
        if best_shot is not None and best_shot[2] >= 9.5:
            note(agent, "shot")
            self._remember_choice(agent, 'shot', best_shot[3], 0, 0)
            Action.kick_one_step(agent, best_shot[1])
            return

        best_cand = candidates[0]
        best_total_score = best_cand[2]

        can_deepen = (
            agent.budget.remaining_ms > 1000 and
            agent.budget.decision_remaining_ms > 60
        )

        if can_deepen:
            top_k = candidates[:min(6, len(candidates))]
            for cand in top_k:
                act_type, target_pt, initial_score, landing_pt, rcv_tm, arr, extra_param = cand
                if act_type in ('pass', 'space_pass', 'dribble') and landing_pt is not None and initial_score > -0.4:
                    ply2_score = self._evaluate_ply2(agent, landing_pt, rcv_tm, opps, teammates)
                    total_score = initial_score + 0.34 * ply2_score
                else:
                    total_score = initial_score

                if total_score > best_total_score:
                    best_total_score = total_score
                    best_cand = cand

        min_viable = -0.40 if ball.x() > -18.0 else -0.05
        if best_total_score > min_viable:
            act_type = best_cand[0]
            extra_param = best_cand[6]

            if act_type == 'shot':
                note(agent, "shot")
                self._remember_choice(agent, 'shot', best_cand[3], 0, 0)
                Action.kick_one_step(agent, best_cand[1])
                return
            elif act_type in ('pass', 'space_pass'):
                note(agent, act_type)
                self._remember_choice(agent, act_type, best_cand[3], cur_pass_streak, 0)
                v_one = self._one_step_kick_speed(agent, best_cand[1])
                if v_one >= 1.75 or cur_pass_streak >= 2 or pressed:
                    Action.kick_one_step(agent, best_cand[1])
                else:
                    Action.kick(agent, best_cand[1], best_cand[5], accept_late=int(extra_param))
                return
            elif act_type == 'dribble':
                note(agent, "dribble")
                self._remember_choice(agent, 'dribble', best_cand[3], 0, cur_drib_streak + 1)
                Action.dribble(agent, best_cand[1])
                return
            elif act_type == 'hold':
                note(agent, "hold")
                self._remember_choice(agent, 'hold', ball, 0, 0, cur_hold_streak + 1)
                Action.hold(agent)
                return

        note(agent, "clear_low_score")
        self._remember_choice(agent, 'clear', ball, 0, 0)
        self._execute_safe_clear(agent, ball, L, W, pressed)

    def _execute_safe_clear(self, agent, ball: Vector2D, L: float, W: float, pressed: bool):
        if ball.x() > 20.0:
            gk = Tools.get_opponent_goalie(agent)
            if gk is not None:
                p_up = Vector2D(L, 6.2)
                p_dn = Vector2D(L, -6.2)
                shot_pt = p_up if gk.position.dist(p_up) > gk.position.dist(p_dn) else p_dn
            else:
                side_y = 6.0 if ball.y() <= 0.0 else -6.0
                shot_pt = Vector2D(L, side_y)
            Action.kick_one_step(agent, shot_pt)
            return
        if not pressed and ball.x() > -10.0:
            Action.advance_ball(agent)
            return
        side = 1.0 if ball.y() >= 0.0 else -1.0
        clear_target = Vector2D(clamp(ball.x() + 42.0, 10.0, L - 4.0),
                                clamp(side * W * 0.72, -W + 3.0, W - 3.0))
        if self._one_step_kick_speed(agent, clear_target) >= 1.85:
            Action.kick_one_step(agent, clear_target)
        else:
            Action.clear(agent)

    # ------------------------------------------------------------------
    # Off-ball coordination & phase logic
    # ------------------------------------------------------------------
    def _get_player_reach_tuple(self, agent, p):
        wm = agent.wm
        bpos = wm.ball.position
        bvel = wm.ball.velocity
        sp = agent.server_params

        if getattr(p, "is_tackling", False):
            return (30, p.position.dist(bpos), p.uniform_number)

        if bvel.r() > 0.45:
            decay = sp.ball_decay
            for n in range(1, 24):
                pt = Tools.inertia_point(bpos, bvel, n, decay)
                c = self._cycles_to_point(agent, p, pt, is_opp=False)
                if c <= n:
                    return (n, p.position.dist(pt), p.uniform_number)
            final_pt = Tools.calculate_ball_inertia_final_point(bpos, bvel, decay)
            d = p.position.dist(final_pt)
            c_final = max(24, self._cycles_to_point(agent, p, final_pt, is_opp=False))
            return (c_final, d, p.uniform_number)

        dist = p.position.dist(bpos)
        cycles = self._cycles_to_point(agent, p, bpos, is_opp=False)
        return (cycles, dist, p.uniform_number)

    def _determine_phase(self, agent, reach_map, outfield_players) -> str:
        wm = agent.wm

        # Goalie controls ball
        if wm.kickable_teammate_existance and wm.kickable_teammate_id == wm.our_goalie_uniform_number:
            return "GOALIE"

        # Opponent controls ball
        if wm.kickable_opponent_existance:
            return "DEFEND"

        # Teammate controls ball or arrives first
        fastest_player = min(outfield_players, key=lambda p: reach_map[p.uniform_number])
        fastest_cyc = reach_map[fastest_player.uniform_number][0]
        t_opp = wm.intercept_table.first_opponent_reach_steps
        if (wm.kickable_teammate_existance and wm.kickable_teammate_id != wm.our_goalie_uniform_number) or \
           (fastest_cyc <= 1 and fastest_cyc <= t_opp):
            return "ATTACK"

        # Pass in flight that we kicked
        last_side = getattr(wm, "last_kick_side", None)
        our_side = getattr(wm, "our_side", None)
        if last_side is not None and our_side is not None and last_side != 0:
            we_kicked_last = (last_side == our_side)
        else:
            we_kicked_last = False
        if we_kicked_last and fastest_cyc <= t_opp + 2:
            return "OUR_PASS"

        return "DEFEND"

    def _solve_role_assignment(self, players, targets):
        # Minimum total distance matching of players to target positions
        if not players or not targets:
            return {}
        best_cost = 1e9
        best_map = {}
        n_t = len(targets)

        def permute(idx, used, cost, cur):
            nonlocal best_cost, best_map
            if idx == len(players) or used == (1 << n_t) - 1:
                if cost < best_cost:
                    best_cost = cost
                    best_map = dict(cur)
                return
            p = players[idx]
            for j in range(n_t):
                if not (used & (1 << j)):
                    pt, lbl = targets[j]
                    c = p.position.dist(pt)
                    cur[p.uniform_number] = (pt, lbl)
                    permute(idx + 1, used | (1 << j), cost + c, cur)
                    del cur[p.uniform_number]

        permute(0, 0, 0.0, {})
        return best_map

    def _build_attack_targets(self, agent, ball: Vector2D, L: float, W: float, offside_x: float):
        wm = agent.wm
        opps = self._get_valid_opponents(wm)
        outfield_opps = [o for o in opps if not o.is_goalie]
        eff_offside = max(0.0, float(offside_x), ball.x())
        x_cap = clamp(eff_offside - 0.85, 0.0, L - 3.0)

        # Build-up in our half
        if ball.x() < -8.0:
            adv_x = clamp(ball.x() + 13.5, -L + 12.0, x_cap)
            left_pt = Vector2D(adv_x, -W * 0.42)
            right_pt = Vector2D(adv_x, W * 0.42)
            for side_sign, default_pt in ((-1.0, left_pt), (1.0, right_pt)):
                best_pt = default_pt
                best_sc = -1e9
                for dy in (5.5, 11.0, 16.5):
                    cy = clamp(side_sign * dy, -W + 3.0, W - 3.0)
                    pt = Vector2D(adv_x, cy)
                    opp_d = min((o.position.dist(pt) for o in outfield_opps), default=20.0)
                    lane_d = min((seg_point_dist(ball.x(), ball.y(), pt.x(), pt.y(),
                                                 o.position.x(), o.position.y())
                                  for o in outfield_opps), default=20.0)
                    if lane_d < 1.3:
                        continue
                    sc = min(opp_d, 10.0) + 0.6 * min(lane_d, 6.0)
                    if sc > best_sc:
                        best_sc = sc
                        best_pt = pt
                if side_sign < 0:
                    left_pt = best_pt
                else:
                    right_pt = best_pt

            piv_x = clamp(ball.x() - 6.0 if ball.x() > -34.0 else ball.x() + 6.5, -L + 9.0, -8.0)
            piv_y = clamp(-ball.y() * 0.35, -8.0, 8.0)
            return [
                (left_pt, "support_build_l"),
                (right_pt, "support_build_r"),
                (Vector2D(piv_x, piv_y), "support_pivot"),
            ]

        # Final third box attack
        if ball.x() > 20.0:
            c1_max = min(x_cap, 44.5)
            c1_x = clamp(max(ball.x() + 5.0, 38.0), 22.0, c1_max) if c1_max >= 22.0 else c1_max
            best_c1 = Vector2D(c1_x, -3.5 if ball.y() >= 0 else 3.5)
            best_c1_sc = -1e9
            for cy in (-4.5, -2.0, 0.0, 2.0, 4.5):
                pt = Vector2D(c1_x, cy)
                opp_d = min((o.position.dist(pt) for o in outfield_opps), default=20.0)
                lane_d = min((seg_point_dist(ball.x(), ball.y(), pt.x(), pt.y(),
                                             o.position.x(), o.position.y())
                              for o in outfield_opps), default=20.0)
                if lane_d < 1.20:
                    continue
                sc = min(opp_d, 9.0) + 0.75 * min(lane_d, 5.0) - 0.08 * abs(cy)
                if sc > best_c1_sc:
                    best_c1_sc = sc
                    best_c1 = pt

            c2_max = min(x_cap, 42.5)
            c2_x = clamp(max(ball.x() + 3.0, 35.0), 20.0, c2_max) if c2_max >= 20.0 else c2_max
            weak_sign = -1.0 if ball.y() >= 0 else 1.0
            best_c2 = Vector2D(c2_x, weak_sign * 6.5)
            best_c2_sc = -1e9
            for dy in (4.5, 7.0, 9.5):
                cy = weak_sign * dy
                pt = Vector2D(c2_x, cy)
                if pt.dist(best_c1) < 5.0:
                    continue
                opp_d = min((o.position.dist(pt) for o in outfield_opps), default=20.0)
                lane_d = min((seg_point_dist(ball.x(), ball.y(), pt.x(), pt.y(),
                                             o.position.x(), o.position.y())
                              for o in outfield_opps), default=20.0)
                if lane_d < 1.20:
                    continue
                sc = min(opp_d, 9.0) + 0.65 * min(lane_d, 5.0)
                if sc > best_c2_sc:
                    best_c2_sc = sc
                    best_c2 = pt

            sweeper_x = clamp(ball.x() - 16.0, -32.0, -6.0)
            sweeper_y = clamp(ball.y() * 0.35, -9.0, 9.0)
            return [
                (best_c1, "support_strike"),
                (best_c2, "support_conn"),
                (Vector2D(sweeper_x, sweeper_y), "support_RestDef"),
            ]

        # Midfield support
        min_fwd = max(2.0, ball.x() + 2.0)
        fwd_x = clamp(ball.x() + 14.0, min_fwd, x_cap) if x_cap >= min_fwd else x_cap
        best_fwd_pt = Vector2D(fwd_x, clamp(-ball.y() * 0.30, -8.0, 8.0))
        best_fwd_sc = -1e9
        for cand_y in (-10.0, -5.0, 0.0, 5.0, 10.0):
            pt = Vector2D(fwd_x, cand_y)
            opp_d = min((o.position.dist(pt) for o in outfield_opps), default=20.0)
            lane_d = min((seg_point_dist(ball.x(), ball.y(), pt.x(), pt.y(),
                                         o.position.x(), o.position.y())
                          for o in outfield_opps), default=20.0)
            lane_pen = 5.5 if lane_d < 1.35 else 0.0
            sc = min(opp_d, 10.0) - lane_pen - 0.12 * abs(cand_y)
            if sc > best_fwd_sc:
                best_fwd_sc = sc
                best_fwd_pt = pt

        best_conn_pt = Vector2D(clamp(ball.x() + 8.0, -4.0, x_cap),
                                clamp(ball.y() + (-9.0 if ball.y() > 0 else 9.0), -W + 4.0, W - 4.0))
        best_conn_sc = -1e9
        for r in (9.0, 14.0):
            for pt in Tools.points_around(ball, r, 12):
                if abs(pt.x()) > L - 2.0 or abs(pt.y()) > 15.0 or pt.x() > x_cap:
                    continue
                if pt.dist(best_fwd_pt) < 6.5:
                    continue
                opp_d = min((o.position.dist(pt) for o in outfield_opps), default=20.0)
                lane_d = min((seg_point_dist(ball.x(), ball.y(), pt.x(), pt.y(),
                                             o.position.x(), o.position.y())
                              for o in outfield_opps), default=20.0)
                if lane_d < 1.4 or opp_d < 2.2:
                    continue
                sc = 0.35 * (pt.x() - ball.x()) + 0.85 * min(opp_d, 10.0) - 0.20 * abs(pt.y())
                if sc > best_conn_sc:
                    best_conn_sc = sc
                    best_conn_pt = pt

        sweeper_x = clamp(ball.x() - 14.0, -36.0, -10.0)
        sweeper_y = clamp(ball.y() * 0.40, -10.0, 10.0)
        sweeper_pt = Vector2D(sweeper_x, sweeper_y)

        return [
            (best_fwd_pt, "support_strike"),
            (best_conn_pt, "support_conn"),
            (sweeper_pt, "support_RestDef"),
        ]

    def _try_universal_tackle(self, agent, wm, ball: Vector2D, L: float, W: float) -> bool:
        if getattr(wm.self, "is_tackling", False):
            return False

        body_deg = float(wm.self.body_direction)

        # Attacking third tackle directed toward net
        if ball.x() > 26.0 and abs(ball.y()) < 14.5:
            if wm.ball.dist_from_self <= 2.15 and wm.self.foul_probability <= 0.22 and wm.self.tackle_probability >= 0.38:
                gk = Tools.get_opponent_goalie(agent)
                gy = -3.8 if (gk is not None and gk.position.y() >= 0.0) else 3.8
                want_w = (Vector2D(L, gy) - ball).th().degree()
                Action.tackle(agent, direction=norm_angle(want_w - body_deg), foul=False)
                note(agent, "tackle_goal")
                return True

        # Defensive and open pitch tackles
        if ball.x() < -22.0:
            our_gk = next((t for t in wm.teammates if t.is_goalie), None)
            if our_gk is not None and our_gk.position.dist(ball) < 8.5 and ball.x() < -32.0:
                return False
            max_foul = 0.12
            if wm.ball.dist_from_self > 1.85 or wm.self.foul_probability > max_foul:
                return False
            min_tp = 0.72 if wm.kickable_opponent_existance else 0.80
            if wm.self.tackle_probability < min_tp:
                return False
        else:
            max_foul = 0.20
            if wm.ball.dist_from_self > 2.15 or wm.self.foul_probability > max_foul:
                return False
            min_tp = 0.42 if wm.kickable_opponent_existance else 0.50
            if wm.self.tackle_probability < min_tp:
                return False

        best_rel = None
        best_sc = -1e9
        for rel in (0.0, -25.0, 25.0, -50.0, 50.0, -80.0, 80.0, 180.0):
            w_ang = norm_angle(body_deg + rel)
            rad = math.radians(w_ang)
            end_x = ball.x() + 12.0 * math.cos(rad)
            end_y = ball.y() + 12.0 * math.sin(rad)
            if end_x < -L + 8.0 and abs(end_y) < 12.5:
                continue
            if ball.x() < -15.0 and math.cos(rad) < -0.25 and abs(end_y) < 15.0:
                continue
            force_sc = 1.0 - (abs(rel) / 180.0) * 0.65
            prog_sc = 0.35 * math.cos(rad)
            sc = force_sc + prog_sc
            if sc > best_sc:
                best_sc = sc
                best_rel = rel

        if best_rel is not None:
            Action.tackle(agent, direction=best_rel, foul=False)
            note(agent, "tackle")
            return True
        return False

    # ------------------------------------------------------------------
    # Off-ball decision making
    # ------------------------------------------------------------------
    def off_ball(self, agent) -> None:
        try:
            self._off_ball_impl(agent)
        except Exception as e:
            note(agent, f"err_{type(e).__name__}")
            Action.go_to_point(agent, Tools.home_position(agent), max_dash_power=self.dash_power(agent))

    def _off_ball_impl(self, agent) -> None:
        wm = agent.wm
        sp = agent.server_params
        L = sp.pitch_half_length
        W = sp.pitch_half_width
        ball = wm.ball.position
        goal = Vector2D(-L, 0.0)

        outfield_players = sorted([wm.self] + self._get_valid_teammates(wm), key=lambda p: p.uniform_number)
        reach_map = {p.uniform_number: self._get_player_reach_tuple(agent, p) for p in outfield_players}
        phase = self._determine_phase(agent, reach_map, outfield_players)

        # Phase 1: Goalkeeper restart / recovery
        if phase == "GOALIE":
            off_idx = next((i for i, p in enumerate(outfield_players)
                            if p.uniform_number == wm.self.uniform_number), 0)
            slots = [
                Vector2D(-44.0, -14.0),
                Vector2D(-44.0, 14.0),
                Vector2D(-38.0, 0.0),
                Vector2D(-24.0, 0.0)
            ]
            target = slots[min(off_idx, len(slots) - 1)]
            note(agent, "phase_goalie")
            Action.go_to_point(agent, target, max_dash_power=self.dash_power(agent))
            return

        # Phase 2: Attacking or pass in flight
        if phase in ("ATTACK", "OUR_PASS"):
            fastest_player = min(outfield_players, key=lambda p: reach_map[p.uniform_number])
            if phase == "OUR_PASS":
                receiver = fastest_player
                if receiver.uniform_number == wm.self.uniform_number:
                    note(agent, "intercept_pass")
                    Action.intercept(agent)
                    return
                active_id = receiver.uniform_number
                n_rcv = clamp(reach_map[active_id][0], 1, 20)
                ref_pos = Tools.inertia_point(ball, wm.ball.velocity, n_rcv, sp.ball_decay)
            else:
                kickable_tm = next((t for t in outfield_players if t.position.dist(ball) < 1.35), None)
                if kickable_tm is not None:
                    active_id = kickable_tm.uniform_number
                elif wm.kickable_teammate_id is not None and wm.kickable_teammate_id > 1:
                    active_id = wm.kickable_teammate_id
                else:
                    active_id = fastest_player.uniform_number
                ref_pos = ball

            support_players = [p for p in outfield_players if p.uniform_number != active_id]
            att_targets = self._build_attack_targets(agent, ref_pos, L, W, wm.offside_line_x)
            assigned = self._solve_role_assignment(support_players, att_targets)
            my_role = assigned.get(wm.self.uniform_number)
            if my_role is not None:
                pt, lbl = my_role
                note(agent, lbl)
                stam = wm.self.stamina
                if stam < 3000:
                    dash_p = 50.0
                elif lbl in ("support_strike", "support_conn") or ref_pos.x() > -5.0:
                    dash_p = 100.0 if stam > 4500 else 76.5
                else:
                    dash_p = self.dash_power(agent)
                Action.go_to_point(agent, pt, max_dash_power=dash_p, distance_threshold=0.7)
                return
            else:
                note(agent, "support_fwd")
                stam = wm.self.stamina
                fwd_dash = 100.0 if stam > 4500 else 76.5
                fwd_pt = Vector2D(clamp(ref_pos.x() + 8.0, -10.0, L - 6.0), clamp(ref_pos.y() * 0.5, -12.0, 12.0))
                Action.go_to_point(agent, fwd_pt, max_dash_power=fwd_dash, distance_threshold=0.7)
                return

        # Phase 3: Defending
        if getattr(wm.self, "is_tackling", False):
            return

        if self._try_universal_tackle(agent, wm, ball, L, W):
            return

        deepest_player = min(outfield_players, key=lambda p: (p.position.dist(goal), p.uniform_number))
        ball_goal_dist = ball.dist(goal)

        def def_press_key(p):
            cyc, dist, unum = reach_map[p.uniform_number]
            if getattr(p, "is_tackling", False):
                cyc += 25
            if (p.position.x() > ball.x() + 0.9 or p.position.dist(goal) > ball_goal_dist + 1.2) and dist > 4.5:
                cyc += 5
            if (ball.x() > -24.0 and p.uniform_number == deepest_player.uniform_number and
                    cyc > wm.intercept_table.first_opponent_reach_steps):
                cyc += 6
            return (cyc, dist, unum)

        def_ranked = sorted(outfield_players, key=def_press_key)
        my_def_rank = next((i for i, p in enumerate(def_ranked)
                            if p.uniform_number == wm.self.uniform_number), 0)

        # Immediate challenge in defensive box
        if (ball.x() < -20.0 and wm.ball.dist_from_self < 3.8) or (ball.x() < -14.0 and wm.ball.dist_from_self < 3.0) or (wm.ball.dist_from_self < 2.3):
            note(agent, "intercept_close")
            Action.intercept(agent)
            return

        # Primary presser
        if my_def_rank == 0:
            note(agent, "intercept")
            Action.intercept(agent)
            return

        # Defensive coverage assignments
        rest_defenders = def_ranked[1:]
        focus_pt = Tools.inertia_point(ball, wm.ball.velocity,
                                       min(6, wm.intercept_table.first_opponent_reach_steps),
                                       sp.ball_decay)
        def_targets = []

        if focus_pt.x() < -20.0:
            # Defensive third compact shape
            goal_anchor = Vector2D(-L, 0.0)
            to_goal = goal_anchor - focus_pt
            if to_goal.r() > 1.0:
                gap = min(3.8, max(2.2, to_goal.r() * 0.20))
                cov_pt = focus_pt + to_goal.set_length_vector(gap)
                cov_x = clamp(cov_pt.x(), -L + 5.5, -12.0)
                cov_y = clamp(cov_pt.y(), -16.5, 16.5)
                def_targets.append((Vector2D(cov_x, cov_y), "cover_center"))

            anchor_x = clamp(focus_pt.x() - 3.8, -L + 6.5, -25.0)
            anchor_y = clamp(focus_pt.y() * 0.25, -6.0, 6.0)
            def_targets.append((Vector2D(anchor_x, anchor_y), "cover_anchor"))

            outfield_opps = sorted([o for o in wm.opponents if o.uniform_number > 0 and not o.is_goalie],
                                   key=lambda o: (o.position.dist(goal), o.position.dist(focus_pt)))
            opps_off_ball = [o for o in outfield_opps if o.position.dist(focus_pt) > 2.0]
            if opps_off_ball:
                opp = opps_off_ball[0]
                opp_pred = opp.position + opp.velocity * 2.0
                v_goal = (goal - opp_pred)
                v_ball = (focus_pt - opp_pred)
                if v_goal.r() > 0.5 and v_ball.r() > 0.5:
                    blend = v_goal.set_length_vector(1.10) + v_ball.set_length_vector(0.55)
                    mark_pt = opp_pred + blend
                else:
                    mark_pt = opp_pred + Vector2D(-1.5, 0.0)
                mx = clamp(mark_pt.x(), -L + 5.0, -14.0)
                my = clamp(mark_pt.y(), -11.0, 11.0)
                def_targets.append((Vector2D(mx, my), "cover_mark"))

            while len(def_targets) < len(rest_defenders):
                sweep_x = clamp(anchor_x - 1.5, -L + 5.5, -28.0)
                sweep_y = clamp(-anchor_y * 0.5, -4.5, 4.5)
                def_targets.append((Vector2D(sweep_x, sweep_y), "cover_sweeper"))
        else:
            # Midfield / high block defense
            target_line_x = max(-26.0, min(-17.0, focus_pt.x() - 5.5))

            our_gk = next((t for t in wm.teammates if t.is_goalie), None)
            open_post_y = 0.0
            goal_anchor = Vector2D(-L, open_post_y)
            to_goal = goal_anchor - focus_pt
            if to_goal.r() > 1.0:
                gap = min(6.0, max(3.3, to_goal.r() * 0.28))
                cov_pt = focus_pt + to_goal.set_length_vector(gap)
                cov_x = clamp(cov_pt.x(), target_line_x, -10.0)
                cov_y = clamp(cov_pt.y() * 0.82, -13.0, 13.0)
                def_targets.append((Vector2D(cov_x, cov_y), "cover_center"))

            outfield_opps = sorted([o for o in wm.opponents if o.uniform_number > 0 and not o.is_goalie],
                                   key=lambda o: o.position.dist(goal))
            opps_off_ball = [o for o in outfield_opps if o.position.dist(focus_pt) > 2.3]

            for m_idx, opp in enumerate(opps_off_ball[:2]):
                opp_pred = opp.position + opp.velocity * 2.0
                v_goal = (goal - opp_pred)
                v_ball = (focus_pt - opp_pred)
                if v_goal.r() > 0.5 and v_ball.r() > 0.5:
                    blend = v_goal.set_length_vector(1.20) + v_ball.set_length_vector(0.65)
                    mark_pt = opp_pred + blend
                else:
                    mark_pt = opp_pred + Vector2D(-1.6, 0.0)

                mx = clamp(mark_pt.x(), target_line_x, L - 5.0)
                my = clamp(mark_pt.y(), -13.5, 13.5)

                if m_idx == 1 and focus_pt.x() > -22.0 and opp_pred.x() > -19.0:
                    mx = clamp(min(mx, focus_pt.x() - 6.5, -18.0), target_line_x, -16.0)
                    my = clamp(my * 0.75, -12.5, 12.5)
                    def_targets.append((Vector2D(mx, my), "cover_sweeper"))
                else:
                    def_targets.append((Vector2D(mx, my), "cover_mark"))

            while len(def_targets) < len(rest_defenders):
                idx_fill = len(def_targets)
                sweep_x = clamp(min(focus_pt.x() - 7.5, -17.0), target_line_x, -16.0)
                side_offset = (-4.5 if focus_pt.y() >= 0 else 4.5) * (1 if idx_fill == 1 else -1)
                sweep_y = clamp(focus_pt.y() * 0.35 + side_offset, -10.0, 10.0)
                def_targets.append((Vector2D(sweep_x, sweep_y), "cover_sweeper"))

        assigned_def = self._solve_role_assignment(rest_defenders, def_targets[:len(rest_defenders)])
        my_def_role = assigned_def.get(wm.self.uniform_number)
        if my_def_role is not None:
            pt, lbl = my_def_role
            stam = wm.self.stamina
            dist_to_target = wm.self.position.dist(pt)
            if stam < 3000:
                dash_p = 50.0
            elif ball.x() < wm.self.position.x() - 1.0 and stam > 3500:
                dash_p = 100.0 if stam > 4500 else 88.0
            elif ball.x() < -20.0 and dist_to_target > 2.5 and stam > 4200:
                dash_p = 90.0
            elif dist_to_target > 4.5 and stam > 4000:
                dash_p = 85.0
            elif ball.x() < 5.0 and stam > 3400:
                dash_p = 76.5
            else:
                dash_p = 65.0
            note(agent, lbl)
            Action.go_to_point(agent, pt, max_dash_power=dash_p, distance_threshold=0.6)
            return

        note(agent, "home")
        Action.go_to_point(agent, Tools.home_position(agent), max_dash_power=self.dash_power(agent))
