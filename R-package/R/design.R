# Design matrices, and their linear map onto the scale the network was trained on.
#
# The designs are built with model.matrix(), so coefficient names, contrasts and
# transformations are those of brms. The network expects standardized inputs, so every
# column is shifted and scaled:
#
#   y* = (y - y_mean) / y_sd          (gaussian models only)
#   x* = (x - x_mean) / x_scale       (population-level columns)
#   z* = (z - z_center) / z_scale     (group-level columns)
#
# x_mean is the column mean when the model has a centered intercept (brms centers all
# columns, dummies included), else 0. Scales are standard deviations, or 1 for 0/1
# columns, which the network saw unscaled in training.
#
# Group-level columns are centered like the population-level ones (z_center = x_mean)
# only for correlated effects with an intercept. There, centering just moves the random
# intercept to the predictor means, and draws are mapped back exactly with the matrix B
# (r = B r_centered). For uncorrelated effects or effects without intercept, centering
# would change the model, so z_center = 0.
#
# Because all maps are linear, priors map to the network's scale and draws map back
# without approximation (see model_hyper() and back_transform()).

# Standard deviation with denominator n, as used by the network's training data.
sd_n <- function(x) sqrt(mean((x - mean(x))^2))

shift_scale <- function(M, shift, scale) {
  unname(sweep(sweep(M, 2, shift), 2, scale, "/"))
}

column_scale <- function(x) {
  if (all(x %in% c(0, 1))) return(1)
  s <- sd_n(x)
  if (s > 1e-12) s else 1
}

# Build everything a fit needs from a parsed formula (see parse_model_formula()).
#
# Returns a list with, among others: frame (the data used), y, X, Z, group indices,
# coefficient names, the scaling constants and `model`, the design on the network's scale.
build_design <- function(spec, data, family, env, drop_unused_levels = TRUE) {
  frame <- model_frame(spec, data, env, drop_unused_levels)
  # `0 + Intercept` refers to a column of ones named Intercept
  eval_frame <- if (spec$int_special) cbind(frame, Intercept = 1) else frame
  likelihood <- family_likelihood(family)
  y <- model_response(spec, eval_frame, family$family, env)

  fixed <- design_block(spec$fixed, eval_frame)
  has_intercept <- "Intercept" %in% fixed$names
  int_type <- if (has_intercept && !spec$int_special && spec$center) "centered" else if (has_intercept) "plain" else "none"
  X <- fixed$M

  random <- random_design(spec, eval_frame, env)
  Z <- random$Z
  groups <- group_index(frame, spec$group_vars)

  # shifts and scales of y, X and Z (an intercept column of ones gets scale 1)
  y_mean <- if (likelihood == "gaussian" && int_type != "none") mean(y) else 0
  y_sd <- if (likelihood == "gaussian") column_scale(y) else 1
  x_scales <- apply(X, 2, column_scale)
  x_means <- colMeans(X)
  x_means[colnames(X) == "Intercept" | int_type != "centered"] <- 0
  z_scales <- apply(Z, 2, column_scale)
  z_centered <- random$correlated && int_type == "centered" && "Intercept" %in% colnames(Z)
  z_centers <- colMeans(Z)
  z_centers[colnames(Z) == "Intercept" | !z_centered] <- 0
  # r = B r_centered moves the random intercept from the predictor means back to x = 0
  B <- diag(ncol(Z))
  dimnames(B) <- list(colnames(Z), colnames(Z))
  if (z_centered) B["Intercept", ] <- B["Intercept", ] - z_centers

  # The network was trained with Z = X[, 1:q], so its X columns start with the intercept
  # and the predictors that also have group-level effects.
  in_z <- match(colnames(Z), colnames(X))
  perm <- unique(c(which(colnames(X) == "Intercept"), in_z[!is.na(in_z)], seq_len(ncol(X))))

  list(
    frame = frame, y = y, X = X, Z = Z,
    groups = groups$codes, levels = groups$levels, observed = groups$observed,
    group = spec$group, group_vars = spec$group_vars,
    fixed_block = fixed[names(fixed) != "M"],
    re_blocks = lapply(random$blocks, function(b) b[names(b) != "M"]),
    likelihood = likelihood, int_type = int_type, int_special = spec$int_special,
    b_names = colnames(X), re_names = colnames(Z), correlated = random$correlated,
    y_mean = y_mean, y_sd = y_sd, x_means = x_means, x_scales = x_scales,
    z_centered = z_centered, z_centers = z_centers, z_scales = z_scales, B = B, perm = perm,
    model = list(
      X = shift_scale(X, x_means, x_scales)[, perm, drop = FALSE],
      Z = shift_scale(Z, z_centers, z_scales),
      y = (y - y_mean) / y_sd,
      groups = groups$codes
    )
  )
}

# The model variables of 'data', without incomplete rows (as in brms).
model_frame <- function(spec, data, env, drop_unused_levels) {
  if (missing(data) || is.null(data)) stop2("Argument 'data' is missing.")
  data <- as.data.frame(data)
  formulas <- c(spec$resp, spec$trials, spec$fixed, lapply(spec$bars, `[[`, "lhs"))
  vars <- setdiff(unique(c(unlist(lapply(formulas, all.vars)), spec$group_vars)), "Intercept")
  missing_vars <- vars[!vars %in% names(data) & !vapply(vars, exists, TRUE, envir = env)]
  if (length(missing_vars)) {
    stop2("The following variables can neither be found in 'data' nor in 'data2':\n", collapse_comma(missing_vars))
  }
  frame <- data[intersect(vars, names(data))]
  complete <- stats::complete.cases(frame)
  if (!all(complete)) {
    warning2("Rows containing NAs were excluded from the model.")
    frame <- frame[complete, , drop = FALSE]
  }
  if (nrow(frame) == 0) stop2("No observations left after removing rows with missing values.")
  if (drop_unused_levels) frame <- droplevels(frame)
  frame
}

# The response, checked against the family (as in brms).
model_response <- function(spec, frame, family, env) {
  y <- eval(spec$resp, frame, env)
  if (family %in% c("bernoulli", "binomial")) {
    if (is.factor(y) && nlevels(y) <= 2) y <- as.numeric(y) - 1  # second level = 1
    if (!all(y %in% c(0, 1))) stop2("Family '", family, "' requires responses to contain only two different values.")
  }
  if (family == "binomial") {
    trials <- if (is.null(spec$trials)) max(y) else eval(spec$trials, frame, env)
    if (is.null(spec$trials)) warning2("Using the maximum response value as the number of trials.")
    if (any(trials != 1)) stop2("metabeta supports binomial models with one trial only. Use brms::brm() for more trials.")
  }
  if (!is.numeric(y) && !is.logical(y)) stop2("Family '", family, "' requires numeric responses.")
  if (family == "poisson") {
    if (any(y != round(y))) stop2("Family 'poisson' requires integer responses.")
    if (any(y < 0)) stop2("Family 'poisson' requires response greater than or equal to 0.")
  }
  as.numeric(y)
}

# model.matrix() of a one-sided formula, plus what is needed to rebuild it for new data.
design_block <- function(formula, frame) {
  mf <- stats::model.frame(formula, data = frame, na.action = stats::na.pass)
  terms <- stats::terms(mf)
  M <- stats::model.matrix(terms, mf)
  raw_names <- colnames(M)
  names <- rename_coef(raw_names)
  if (anyDuplicated(names)) {
    stop2("Coefficient names ", collapse_comma(raw_names), " coincide after renaming. Rename the variables.")
  }
  colnames(M) <- names
  list(
    M = M, names = names, raw_names = raw_names,
    terms = terms, xlevels = stats::.getXlevels(terms, mf), contrasts = attr(M, "contrasts")
  )
}

# The group-level design: one block per term `(lhs | g)`, combined into Z.
random_design <- function(spec, frame, env) {
  blocks <- lapply(spec$bars, function(bar) {
    block <- design_block(stats::as.formula(call("~", bar$lhs), env = env), frame)
    block$correlated <- bar$correlated
    block$id <- bar$id
    block
  })
  Z <- do.call(cbind, lapply(blocks, `[[`, "M"))
  if (anyDuplicated(colnames(Z))) {
    stop2("Duplicated group-level effects are not allowed: ", collapse_comma(colnames(Z)[duplicated(colnames(Z))]), ".")
  }
  list(Z = Z, blocks = blocks, correlated = is_correlated(blocks, ncol(Z)))
}

# metabeta estimates all correlations between the group-level effects or none.
# Correlated: a single correlated block. Uncorrelated: blocks of size one or with `||`.
is_correlated <- function(blocks, q) {
  if (q == 1) return(FALSE)
  # terms with the same ID form one block; every other term is a block of its own
  ids <- vapply(seq_along(blocks), function(i) if (is.na(blocks[[i]]$id)) paste0(".", i) else blocks[[i]]$id, "")
  size <- tapply(vapply(blocks, function(b) length(b$names), 1L), ids, sum)
  correlated <- tapply(vapply(blocks, `[[`, TRUE, "correlated"), ids, all)
  if (length(size) == 1 && correlated[[1]]) return(TRUE)
  if (all(size == 1 | !correlated)) return(FALSE)
  stop2(
    "metabeta estimates either all correlations between the group-level effects of a ",
    "grouping factor or none. Combine the terms into one, e.g. '(1 + x | g)', or make ",
    "them uncorrelated, e.g. '(1 + x || g)'."
  )
}

# Labels of the grouping factor; interactions g1:g2 become "a_b" (as in brms).
group_labels <- function(frame, group_vars) {
  do.call(paste, c(lapply(group_vars, function(v) as.character(frame[[v]])), sep = "_"))
}

# Levels of the grouping factor, the levels with data ('observed'), and 0-based codes
# of each row among the observed levels (the network's group indices).
group_index <- function(frame, group_vars) {
  labels <- group_labels(frame, group_vars)
  # brms: levels of a factor as given, numbers sorted numerically, combinations as text
  levels <- if (length(group_vars) == 1) levels(as.factor(frame[[group_vars]])) else levels(factor(labels))
  level <- match(labels, levels)
  observed <- sort(unique(level))
  list(levels = levels, observed = observed, codes = match(level, observed) - 1L)
}

# Designs for new data: X, Z and each row's index among the fitted levels, with new
# levels numbered after them (their labels are in attribute "new_levels").
design_newdata <- function(design, newdata, need_groups = TRUE, allow_new_levels = FALSE) {
  newdata <- as.data.frame(newdata)
  if (design$int_special) newdata$Intercept <- 1
  blocks <- c(list(design$fixed_block), design$re_blocks)
  vars <- unique(unlist(lapply(blocks, function(b) all.vars(b$terms))))
  vars <- setdiff(c(vars, if (need_groups) design$group_vars), "Intercept")
  missing_vars <- setdiff(vars, names(newdata))
  if (length(missing_vars)) {
    stop2("The following variables can neither be found in 'data' nor in 'data2':\n", collapse_comma(missing_vars))
  }
  X <- rebuild_block(design$fixed_block, newdata)
  Z <- do.call(cbind, lapply(design$re_blocks, rebuild_block, newdata = newdata))
  index <- rep(NA_integer_, nrow(newdata))
  if (need_groups) {
    labels <- group_labels(newdata, design$group_vars)
    index <- match(labels, design$levels)
    new_levels <- unique(labels[is.na(index)])
    if (length(new_levels) && !allow_new_levels) {
      stop2(
        "Levels ", collapse_comma(new_levels), " of grouping factor '", design$group, "' cannot be ",
        "found in the fitted model. Consider setting argument 'allow_new_levels' to TRUE."
      )
    }
    index[is.na(index)] <- length(design$levels) + match(labels[is.na(index)], new_levels)
    attr(index, "new_levels") <- new_levels
  }
  list(X = X, Z = Z, index = index)
}

rebuild_block <- function(block, newdata) {
  mf <- stats::model.frame(block$terms, newdata, xlev = block$xlevels, na.action = stats::na.pass)
  M <- stats::model.matrix(block$terms, mf, contrasts.arg = block$contrasts)
  if (!identical(colnames(M), block$raw_names)) stop2("The design for 'newdata' does not match the fitted model.")
  unname(M)
}
